import argparse
import os
import random
import time

import numpy as np
import torch

from dataloaders.data_dataloaders import DATALOADER_DICT
from metrics import compute_metrics
from metrics import tensor_text_to_video_metrics
from metrics import tensor_video_to_text_sim
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4IDC
from modules.optimization import BertAdam
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

from exploringDebugging import write_debug

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

if torch.cuda.device_count() > 1:
    torch.distributed.init_process_group(backend="nccl")


# 1. Force the backend to 'agg' by OVERRIDING any existing value
#    This MUST be done BEFORE importing matplotlib
os.environ["MPLBACKEND"] = "agg"

# 2. Now import matplotlib
import matplotlib

# 3. (Optional but recommended) Explicitly tell matplotlib to use 'agg'
#    This ensures ist's set, even if another library tried to import it first.
try:
    matplotlib.use("agg")
except Exception:
    pass # Handle potential errors if already set

# 4. Import pyplot
import matplotlib.pyplot as plt

global logger

jsonPath = ""

def get_args(description="CLIP4IDC on Retrieval Task"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.") 
    parser.add_argument("--do_retrieval", action="store_true")

    parser.add_argument("--data_path", type=str, default="data/datatype", help="data file path")
    parser.add_argument("--features_path", type=str, default="data/datatype/images", help="feature path")

    parser.add_argument("--json_path", type=str, default="", help="merged json path")

    parser.add_argument("--num_thread_reader", type=int, default=1, help="")
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--batch_size_val", type=int, default=64, help="batch size eval")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate exp epoch decay")
    parser.add_argument("--n_display", type=int, default=100, help="Information display frequence")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_words", type=int, default=20, help="")
    parser.add_argument("--feature_framerate", type=int, default=1, help="")
    parser.add_argument("--margin", type=float, default=0.1, help="margin for loss")
    parser.add_argument("--hard_negative_rate", type=float, default=0.5, help="rate of intra negative sample")
    parser.add_argument("--negative_weighting", type=int, default=1, help="Weight the loss for intra negative")
    parser.add_argument("--n_pair", type=int, default=1, help="Num of pair to output from data loader")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and " "checkpoints will be written.",
    )
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--resume_model_opt", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup " "for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a " "backward/update pass.",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded " "from s3",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) " "instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in "
        "['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument("--coef_lr", type=float, default=1.0, help="coefficient for bert branch.")
    parser.add_argument("--use_mil", action="store_true", help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument("--sampled_use_mil", action="store_true", help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument("--text_num_hidden_layers", type=int, default=12, help="Layer NO. of text.")
    parser.add_argument("--visual_num_hidden_layers", type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument("--intra_num_hidden_layers", type=int, default=9, help="Layer NO. of intra module")
    parser.add_argument("--cross_num_hidden_layers", type=int, default=2, help="Layer NO. of cross.")

    parser.add_argument("--freeze_layer_num", type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument(
        "--linear_patch", type=str, default="2d", choices=["2d", "3d"], help="linear projection of flattened patches."
    )

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    args = parser.parse_args()

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: " f"{args.gradient_accumulation_steps}, should be >= 1",
        )
    if not args.do_train and not args.do_eval and not args.do_retrieval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` or `do_retrieval` must be True.",
        )

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    jsonPath = args.json_path

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Inside set_seed_logger(args)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    args.world_size = world_size
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
    else:
        local_rank = 0
    torch.cuda.set_device(args.local_rank)
    args.rank = local_rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu', weights_only=True)
    elif args.resume_model:
        model_state_dict = torch.load(args.resume_model, map_location='cpu', weights_only=True)
        logger.info("✅ Resume model state loaded successfully.")
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4IDC.from_pretrained(args.cross_model, args.decoder_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp
                           if "clip." in n
                           and "clip.visual.ln_mid" not in n
                           and "clip.visual.joint_positional_embedding" not in n
                           and "clip.visual.bef_embedding" not in n
                           and "clip.visual.aft_embedding" not in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp
                             if "clip.visual.ln_mid" in n
                             or "clip.visual.joint_positional_embedding" in n
                             or "clip.visual.bef_embedding" in n
                             or "clip.visual.aft_embedding" in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp
                              if "clip." in n
                              and "clip.visual.ln_mid" not in n
                              and "clip.visual.joint_positional_embedding" not in n
                              and "clip.visual.bef_embedding" not in n
                              and "clip.visual.aft_embedding" not in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp
                                if "clip.visual.ln_mid" in n
                                or "clip.visual.joint_positional_embedding" in n
                                or "clip.visual.bef_embedding" in n
                                or "clip.visual.aft_embedding" in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file


def load_model(epoch, args, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(
            args.output_dir,
            f"pytorch_model.bin.{epoch}",
        )
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location="cpu")

        logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = (
            args.cache_dir
            if args.cache_dir
            else os.path.join(
                str(PYTORCH_PRETRAINED_BERT_CACHE),
                "distributed",
            )
        )
        model = CLIP4IDC.from_pretrained(
            args.cross_model,
            cache_dir=cache_dir,
            state_dict=model_state_dict,
            task_config=args,
        )

        model.to(device)
    else:
        model = None
    return model


def train_epoch(
    epoch,
    args,
    model,
    train_dataloader,
    device,
    n_gpu,
    optimizer,
    scheduler,
    global_step,
    local_rank
):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        try:
            # Verileri tek GPU'ya taşı
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
                bef_image,
                aft_image,
                bef_semantic,
                aft_semantic,
                image_mask,
            ) = batch

            #logger.warning("<"*10+"inferencing")

            loss = model(
                input_ids,
                segment_ids,
                input_mask,
                bef_image,
                aft_image,
                bef_semantic,
                aft_semantic,
                image_mask,
            )

            #logger.warning("<"*10+"inferenced")
            #logger.warning("<"*10+str(loss))


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.info(f"HATA: {name} gradyanında NaN veya Inf bulundu!")


            loss.backward()

            #logger.warning("loss backward ")

            total_loss += float(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                #logger.warning("printing epoch info ")


                if scheduler is not None:
                    scheduler.step()  # Update learning rate schedule

                optimizer.step()
                optimizer.zero_grad()

                # Clamp logit scale
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

                #logger.warning("torch.clamp operation done ")

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

        except Exception as e:
            logger.error(f"Error at step {step}: {str(e)}")
            raise

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(
    model,
    batch_list_t,
    batch_list_v,
    batch_sequence_output_list,
    batch_visual_output_list,
):
    sim_matrix = []
    write_debug("batch run on singledaki", batch_list_t, False)
    write_debug("batch_sequence_output_list", batch_sequence_output_list, False)
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            pair_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(
                sequence_output,
                visual_output,
                input_mask,
                pair_mask,
            )
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def _run_on_single_gpu_retrieval(
    model,
    index
):
    # Dosya yollarını kendine göre düzenle
    okuyucu = VeriSetiOkuyucu(
        tensor_path='tum_veri_seti_birlestirilmis.pt', 
        json_path='/content/CLIP4IDC/Second_CC_dataset/SECOND-CC-AUG/merged.json'
    )

    data = okuyucu.get_item(index)

    input_mask = data["input_mask"]
    segment_ids = data["segment_ids"]
    sequence_output = data["sequence_output"]

    result = []

    device = next(model.parameters()).device

    for i, c_visual_output in enumerate(okuyucu.visual_output):
        pair_mask = okuyucu.get_item(i)["pair_mask"]
        b1b2_logits, *_tmp = model.get_similarity_logits(
            sequence_output.to(device).unsqueeze(0),
            c_visual_output.to(device).unsqueeze(0),
            input_mask,
            pair_mask,
        )
        b1b2_logits = b1b2_logits.cpu().detach().numpy()
        result.append(b1b2_logits)

    return result


def eval_epoch(args, model, test_dataloader, device):
    if hasattr(model, "module"):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    # below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, pair_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, "multi_sentence_per_pair") and test_dataloader.dataset.multi_sentence_per_pair:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        pair_num_ = test_dataloader.dataset.image_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per pair setting.")
        logger.warning(f"sentence num: {sentence_num_}, pair num: {pair_num_}")

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_pair_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        write_debug("test dataloader", test_dataloader, False)
        write_debug("data set test dataloader'in içindeki", test_dataloader.dataset, False)
        dontLoop = True
        for bid, batch in enumerate(test_dataloader):
            write_debug("length of batch", len(batch[0]), dontLoop)
            dontLoop = False
            batch = tuple(t.to(device) for t in batch)
            
            (
                input_ids,
                input_mask,
                segment_ids,
                bef_image,
                aft_image,
                bef_semantic,
                aft_semantic,
                image_mask,
            ) = batch

            image_pair = torch.cat([bef_image, aft_image], 1)
            semantic_pair = torch.cat([bef_semantic, aft_semantic], 1)

            if multi_sentence_:
                # multi-sentences retrieval means: one pair has two or more
                # descriptions.
                b, *_t = image_pair.shape
                sequence_output, _ = model.get_sequence_output(
                    input_ids,
                    segment_ids,
                    input_mask,
                )

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append(
                    (
                        input_mask,
                        segment_ids,
                    ),
                )

                s_, e_ = total_pair_num, total_pair_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    image_pair, pair_mask = (
                        image_pair[filter_inds, ...],
                        image_mask[filter_inds, ...],
                    )

                    semantic_pair, pair_mask = (
                        semantic_pair[filter_inds, ...],
                        image_mask[filter_inds, ...],
                    )
                    visual_output, _ = model.get_visual_output(
                        image_pair,
                        semantic_pair,
                        pair_mask,
                    )

                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((pair_mask,))
                total_pair_num += b

            logger.info(f"{bid}/{len(test_dataloader)}\r")
            #print(f"{bid}/{len(test_dataloader)}\r", end="", flush=True)


        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------

        sim_matrix = _run_on_single_gpu(
            model,
            batch_list_t,
            batch_list_v,
            batch_sequence_output_list,
            batch_visual_output_list,
        )
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    if multi_sentence_:
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max(
            [
                e_ - s_
                for s_, e_ in zip(
                    [0] + cut_off_points2len_[:-1],
                    cut_off_points2len_,
                )
            ],
        )
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(
                np.concatenate(
                    (
                        sim_matrix[s_:e_],
                        np.full(
                            (max_length - e_ + s_, sim_matrix.shape[1]),
                            -np.inf,
                        ),
                    ),
                    axis=0,
                ),
            )
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info(
            "after reshape, sim matrix size: {} x {} x {}".format(
                sim_matrix.shape[0],
                sim_matrix.shape[1],
                sim_matrix.shape[2],
            ),
        )

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))

    logger.info("Text-to-Image-Pair:")
    logger.info(
        "\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - "
        "Mean R: {:.1f}".format(
            tv_metrics["R1"],
            tv_metrics["R5"],
            tv_metrics["R10"],
            tv_metrics["MR"],
            tv_metrics["MeanR"],
        ),
    )
    logger.info("Image-Pair-to-Text:")
    logger.info(
        "\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - "
        "V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}".format(
            vt_metrics["R1"],
            vt_metrics["R5"],
            vt_metrics["R10"],
            vt_metrics["MR"],
            vt_metrics["MeanR"],
        ),
    )



    # ... keep all your existing code up to sim_matrix creation ...

    
    # ----------------------------------
    # 3. Get Top-5 Most Similar Sentences for a Random Image Pair
    # ----------------------------------
        # ----------------------------------
    # 3. Get Top-5 Most Similar Sentences for a Random Image Pair
    # ----------------------------------
        # ----------------------------------
    # 3. Get Top-5 Most Similar Sentences for a Random Image Pair (works for multi-sentence sim_matrix)
    # ----------------------------------
    import random

    # If dataset stores the raw sentences/captions, use them.
    # Expect length == total_sentence_num (e.g. 6121 in your logs).
    if hasattr(test_dataloader.dataset, "texts"):
        text_list = test_dataloader.dataset.texts
    else:
        # fallback: create labels for the flattened sentences
        # NOTE: when multi_sentence_, total_sentence_num should equal sim_matrix.shape[0]*sim_matrix.shape[1]
        total_sentences = sim_matrix.shape[0] * sim_matrix.shape[1]
        text_list = [f"Sentence {i}" for i in range(total_sentences)]

    topk = 5

    # Flatten first two dims (pair, sentence_per_pair) -> sentences
    # sim_matrix shape is (pair_num, max_sentences, pair_num) => we want (pair_num * max_sentences, pair_num)
    sim_flat = sim_matrix.reshape(-1, sim_matrix.shape[2])  # shape: (num_sentences, num_image_pairs)

    # pick a random image pair index
    random_img_idx = random.randint(0, sim_flat.shape[1] - 1)

    # get similarities for every sentence to that image pair (1D array)
    sim_for_image = sim_flat[:, random_img_idx]  # shape: (num_sentences,)

    # get topk sentence indices (descending)
    topk_indices = np.argsort(-sim_for_image)[:topk]

    # ensure Python list of ints
    topk_indices = np.array(topk_indices).reshape(-1).astype(int).tolist()

    # fetch sentences and scalar scores
    topk_texts = [text_list[i] for i in topk_indices]
    topk_scores = [float(sim_for_image[i]) for i in topk_indices]

    # print results
    print(f"\n  Randomly Selected Image Pair Index: {random_img_idx}")
    print("Top-5 Most Similar Sentences:")
    for rank, (sent, score) in enumerate(zip(topk_texts, topk_scores), 1):
        print(f"  {rank}. {sent} (score={score:.4f})")

    R1 = tv_metrics["R1"]
    return R1

def eval_epoch(args, model, test_dataloader, device):
    if hasattr(model, "module"):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    # below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, pair_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, "multi_sentence_per_pair") and test_dataloader.dataset.multi_sentence_per_pair:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        pair_num_ = test_dataloader.dataset.image_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per pair setting.")
        logger.warning(f"sentence num: {sentence_num_}, pair num: {pair_num_}")

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_pair_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------

        # 1. Kayıtlar için temiz bir klasör açalım
        output_folder = "batch_kayitlari"
        os.makedirs(output_folder, exist_ok=True)

        print("İşlem başlıyor, veriler parça parça diske yazılacak...")

        for bid, batch in enumerate(test_dataloader):
            logger.warning("Flag!!!!")
            batch = tuple(t.to(device) for t in batch)
            
            (
                input_ids,
                input_mask,
                segment_ids,
                bef_image,
                aft_image,
                bef_semantic,
                aft_semantic,
                image_mask,
            ) = batch

            image_pair = torch.cat([bef_image, aft_image], 1)
            semantic_pair = torch.cat([bef_semantic, aft_semantic], 1)

            if multi_sentence_:
                # multi-sentences retrieval means: one pair has two or more
                # descriptions.
                b, *_t = image_pair.shape
                sequence_output, _ = model.get_sequence_output(
                    input_ids,
                    segment_ids,
                    input_mask,
                )

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append(
                    (
                        input_mask,
                        segment_ids,
                    ),
                )

                s_, e_ = total_pair_num, total_pair_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    image_pair, pair_mask = (
                        image_pair[filter_inds, ...],
                        image_mask[filter_inds, ...],
                    )

                    semantic_pair, pair_mask = (
                        semantic_pair[filter_inds, ...],
                        image_mask[filter_inds, ...],
                    )
                    visual_output, _ = model.get_visual_output(
                        image_pair,
                        semantic_pair,
                        pair_mask,
                    )

                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((pair_mask,))
                total_pair_num += b

                batch_data = {
                    'visual_output': visual_output.detach().cpu(),
                    'sequence_output': sequence_output.detach().cpu(),
                    'input_mask': input_mask.detach().cpu(),
                    'segment_ids': segment_ids.detach().cpu(),
                    'pair_mask': pair_mask.detach().cpu(),
                }
                
                # Her batch için benzersiz bir isim: batch_0.pt, batch_1.pt...
                file_name = os.path.join(output_folder, f"batch_{bid}.pt")
                torch.save(batch_data, file_name)

                # İstersen takip için log bas
                if bid % 10 == 0:
                    print(f"Batch {bid} kaydedildi.")

            # Loading saved
            #veri = torch.load('model_cikti_verileri.pt')
            #embeddings = veri['embeddings']

            logger.info(f"{bid}/{len(test_dataloader)}\r")
            #print(f"{bid}/{len(test_dataloader)}\r", end="", flush=True)


        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------

        sim_matrix = _run_on_single_gpu(
            model,
            batch_list_t,
            batch_list_v,
            batch_sequence_output_list,
            batch_visual_output_list,
        )
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    if multi_sentence_:
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max(
            [
                e_ - s_
                for s_, e_ in zip(
                    [0] + cut_off_points2len_[:-1],
                    cut_off_points2len_,
                )
            ],
        )
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(
                np.concatenate(
                    (
                        sim_matrix[s_:e_],
                        np.full(
                            (max_length - e_ + s_, sim_matrix.shape[1]),
                            -np.inf,
                        ),
                    ),
                    axis=0,
                ),
            )
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info(
            "after reshape, sim matrix size: {} x {} x {}".format(
                sim_matrix.shape[0],
                sim_matrix.shape[1],
                sim_matrix.shape[2],
            ),
        )

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))

    logger.info("Text-to-Image-Pair:")
    logger.info(
        "\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - "
        "Mean R: {:.1f}".format(
            tv_metrics["R1"],
            tv_metrics["R5"],
            tv_metrics["R10"],
            tv_metrics["MR"],
            tv_metrics["MeanR"],
        ),
    )
    logger.info("Image-Pair-to-Text:")
    logger.info(
        "\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - "
        "V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}".format(
            vt_metrics["R1"],
            vt_metrics["R5"],
            vt_metrics["R10"],
            vt_metrics["MR"],
            vt_metrics["MeanR"],
        ),
    )

    R1 = tv_metrics["R1"]
    return R1

import json

class VeriSetiOkuyucu:
    def __init__(self, tensor_path, json_path, split='test'):
        print("Veriler yükleniyor, lütfen bekleyin...")
        
        # 1. Matematiksel Veriyi (Tensors) Yükle
        # (CPU'ya map ederek yüklüyoruz ki GPU dolmasın)

        self.tensors_data = torch.load(tensor_path, map_location=torch.device('cpu'), mmap=True)

        self.visual_output = self.tensors_data['visual_output']
        self.input_mask = self.tensors_data['input_mask']
        self.sequence_output = self.tensors_data['sequence_output']
        self.segment_ids = self.tensors_data['segment_ids']
        self.pair_mask = self.tensors_data['pair_mask']
        
        # 2. Ham Metin Verisini (JSON) Yükle
        with open(json_path, 'r') as f:
            full_json = json.load(f)
            
        # 3. Sadece 'test' kısmını filtrele ve Düzleştir (Flatten)
        # Modelin 6121 satırlık çıktı verdiği için JSON'u da ona benzetmeliyiz.
        self.metadata_list = []
        
        for img_item in full_json['images']:
            if img_item['split'] == split:
                img_filename = img_item['filename']
                img_id = img_item['imgid']
                
                # Her resmin içindeki cümleleri tek tek listeye ekle
                for sent in img_item['sentences']:
                    self.metadata_list.append({
                        'image_filename': img_filename,
                        'image_id': img_id,
                        'raw_text': sent['raw'],
                        'tokens': sent['tokens']
                    })
        
        # Güvenlik Kontrolü
        print(f"Tensor Satır Sayısı: {len(self.sequence_output)}")
        print(f"Metin Satır Sayısı:  {len(self.metadata_list)}")
        
        if len(self.sequence_output) != len(self.metadata_list):
            print("UYARI: Tensor ve Metin sayıları uyuşmuyor! Veri setinde eksik cümleler olabilir.")

    def get_item(self, index):
        """İstenilen sıradaki verinin hem metnini hem tensörünü getirir."""
        if index >= len(self.metadata_list):
            return "Hata: Geçersiz indeks!"
            
        meta = self.metadata_list[index]

        visual_output = self.visual_output[index]
        segment_ids = self.segment_ids[index]
        input_mask = self.input_mask[index]
        sequence_output = self.sequence_output[index]
        pair_mask = self.pair_mask[index]

        
        return {
            'text': meta['raw_text'],
            'image_file': meta['image_filename'],

            'visual_output': visual_output,  
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'sequence_output': sequence_output,
            'pair_mask': pair_mask
        }

def accumulate_vector():
    import glob

    # Kayıtlı tüm dosyaları bul (Sıralı olması için sort kullanıyoruz)
    files = sorted(glob.glob(os.path.join("batch_kayitlari", "*.pt")))

    all_visual_output = []
    all_input_mask = []
    all_segment_ids = []
    all_sequence_output= []
    all_pair_mask = []

    print("Dosyalar okunuyor ve birleştiriliyor...")

    for file_path in files:
        # 1. Dosyayı yükle
        data = torch.load(file_path)
        
        # 2. Listelere ekle
        all_visual_output.append(data['visual_output'])
        all_input_mask.append(data['input_mask'])
        all_segment_ids.append(data['segment_ids'])
        all_sequence_output.append(data['sequence_output'])
        all_pair_mask.append(data['pair_mask'])

    # 3. Hepsini tek bir büyük Tensör yap (Concatenate)
    final_visual_output = torch.cat(all_visual_output, dim=0)
    final_input_mask = torch.cat(all_input_mask, dim=0)
    final_segment_ids = torch.cat(all_segment_ids, dim=0)
    final_sequence_output = torch.cat(all_sequence_output, dim=0)
    final_pair_mask = torch.cat(all_pair_mask, dim=0)
    

    print("BİRLEŞTİRME TAMAMLANDI!")
    print(f"Final Embedding Boyutu: {final_sequence_output.shape}")
    # Örn Çıktı: (6121, 77, 512)

    # İstersen bu BÜYÜK birleşmiş halini de tek dosya olarak saklayabilirsin
    torch.save({
        'visual_output': final_visual_output.detach().cpu(),
        'input_mask': final_input_mask.detach().cpu(),
        'segment_ids': final_segment_ids.detach().cpu(),
        'sequence_output': final_sequence_output.detach().cpu(),
        'pair_mask': final_pair_mask.detach().cpu()
    }, "tum_veri_seti_birlestirilmis.pt")



def print_topk_texts(topk_indices, test_dataloader):
    if hasattr(test_dataloader.dataset, "texts"):
        text_list = test_dataloader.dataset.texts
    else:
        text_list = [f"Sentence {i}" for i in range(len(topk_indices))]

    for i, idx_list in enumerate(topk_indices):
        print(f"\n🖼 Image Pair {i}:")
        for rank, idx in enumerate(idx_list, start=1):
            print(f"  {rank}. {text_list[idx]}")

def _get_clip_projection_dims(model):
    """
    Model içinde varsa clip text/visual projection katmanlarını döndürür.
    Dönen tuple: (has_text_proj, text_proj_tensor, has_visual_proj, visual_proj_tensor)
    text_proj_tensor shape: (D_text, D_embed)  (torch.Tensor or None)
    visual_proj_tensor shape: (D_vis, D_embed) (torch.Tensor or None)
    """
    text_proj = None
    vis_proj = None
    has_text = False
    has_vis = False

    if hasattr(model, "clip"):
        clip = model.clip
        # Common names used in CLIP-like models
        for name in ("text_projection", "text_proj", "proj_text", "text_proj.weight"):
            if hasattr(clip, name):
                text_proj = getattr(clip, name)
                has_text = True
                break
        # If it's a parameter inside nn.Module (e.g. clip.text_projection is nn.Parameter)
        if not has_text:
            # try to find attribute that endswith text_projection
            for n, p in getattr(clip, "named_parameters", lambda: [])():
                if "text_projection" in n or "text_proj" in n:
                    text_proj = p
                    has_text = True
                    break

        for name in ("visual_projection", "visual_proj", "proj_visual", "visual_projection.weight"):
            if hasattr(clip, name):
                vis_proj = getattr(clip, name)
                has_vis = True
                break
        if not has_vis:
            for n, p in getattr(clip, "named_parameters", lambda: [])():
                if "visual_projection" in n or "visual_proj" in n:
                    vis_proj = p
                    has_vis = True
                    break

    return has_text, text_proj, has_vis, vis_proj

def _apply_projection(vec: torch.Tensor, proj):
    """
    vec: (B, D_in)
    proj: can be nn.Parameter or nn.Module or numpy array -> return vec @ proj (B, D_out)
    """
    if proj is None:
        return vec
    if isinstance(proj, torch.nn.Parameter) or isinstance(proj, torch.Tensor):
        # proj shape expected (D_in, D_out) or (D_out, D_in) — try to detect
        p = proj
        if p.ndim == 1:
            # unexpected, just return
            return vec
        if p.shape[0] == vec.shape[1]:
            # (D_in, D_out)
            return vec @ p.to(vec.dtype).to(vec.device)
        elif p.shape[1] == vec.shape[1]:
            # (D_out, D_in) -> do vec @ p.T
            return vec @ p.T.to(vec.dtype).to(vec.device)
        else:
            # shapes incompatible
            raise RuntimeError(f"Projection weight shape {tuple(p.shape)} incompatible with vec dim {vec.shape[1]}")
    elif hasattr(proj, "forward"):
        return proj(vec)
    else:
        return vec

def save_text_embeddings(model, test_dataloader, device, save_path="text_embeddings.npy"):
    import numpy as np
    if hasattr(model, "module"):
        model = model.module

    model.eval()
    all_text_embeddings = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, *_ = batch

            sequence_output, _ = model.get_sequence_output(
                input_ids,
                segment_ids,
                input_mask,
            )

            # ✅ Eğer çıktı (B, D) ise → zaten pooled → direk kullan
            if sequence_output.dim() == 2:
                text_emb = sequence_output  # (B, D)

            # ✅ Eğer çıktı (B, L, D) ise → normal pooling
            elif sequence_output.dim() == 3:
                mask = input_mask.unsqueeze(-1).float()  # (B, L, 1)
                sequence_output = sequence_output * mask
                text_emb = sequence_output.sum(dim=1) / mask.sum(dim=1)

            # ✅ Eğer çıktı (B, D, L) ise → önce transpose → sonra pooling
            elif sequence_output.dim() == 3 and sequence_output.shape[1] != input_mask.shape[1]:
                sequence_output = sequence_output.transpose(1, 2)  # (B, L, D)
                mask = input_mask.unsqueeze(-1).float()
                sequence_output = sequence_output * mask
                text_emb = sequence_output.sum(dim=1) / mask.sum(dim=1)

            else:
                raise ValueError(f"Beklenmeyen text embedding şekli: {sequence_output.shape}")

            all_text_embeddings.append(text_emb.cpu().numpy())

    all_text_embeddings = np.vstack(all_text_embeddings)
    np.save(save_path, all_text_embeddings)
    print(f"✅ Text embeddings saved to {save_path}")

def find_topk_from_saved_text(model, image_pair_batch, device, test_dataloader, embeddings_path="text_embeddings.npy", topk=5):
    """
    Kaydedilmiş text embeddinglerle verilen image_pair_batch için top-k textleri döndürür.
    image_pair_batch: (bef_image, aft_image, bef_semantic, aft_semantic, image_mask)
    """
    if hasattr(model, "module"):
        model = model.module

    if not os.path.exists(embeddings_path):
        print("Dosya yok kayıt alınıyor...")
        save_text_embeddings(model, test_dataloader, device, embeddings_path)

    text_embeddings = np.load(embeddings_path)  # (N, D_text_emb)
    # convert to torch
    text_embeddings_torch = torch.tensor(text_embeddings, device=device)  # (N, Dt)

    # detect projeksiyonlar
    has_text_proj, text_proj, has_vis_proj, vis_proj = _get_clip_projection_dims(model)

    model.eval()
    with torch.no_grad():
        bef_image, aft_image, bef_semantic, aft_semantic, image_mask = image_pair_batch

        

        bef_img = bef_image[0].detach().cpu().permute(1,2,0).numpy()
        aft_img = aft_image[0].detach().cpu().permute(1,2,0).numpy()

        if bef_img.min() < 0 or bef_img.max() > 1:
            bef_img = (bef_img - bef_img.min()) / (bef_img.max() - bef_img.min())
            aft_img = (aft_img - aft_img.min()) / (aft_img.max() - aft_img.min())

        save_path = f"/content/CLIP4IDC/output/preview_{i}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("✅ Saved preview image to:", save_path)
        plt.close()

        # ------------------------------------


        image_pair = torch.cat([bef_image, aft_image], 1)
        semantic_pair = torch.cat([bef_semantic, aft_semantic], 1)

        # image_embedding: (B, T_frames, D_vis)  — kullandığınız modelde T_frames=1 olabilir

        image_embedding, _ = model.get_visual_output(image_pair, semantic_pair, image_mask)

        # 🔥 EKLE → embed’in kaç boyutlu olduğuna bak
        if image_embedding.dim() == 3:
            # (B, T, D) → videoda frame ortalaması
            image_embedding = image_embedding.mean(dim=1)
        elif image_embedding.dim() == 2:
            # (B, D) → hiçbir şey yapma
            pass
        elif image_embedding.dim() == 1:
            # (D,) → batch boyutu yok → batch=1 gibi davran
            image_embedding = image_embedding.unsqueeze(0)
        else:
            raise RuntimeError(f"Beklenmeyen image embedding shape: {image_embedding.shape}")


        # Eğer modelde visual proj varsa onu uygula ki D_vis -> D_embed olsun (text_embeddings ile eşleşecek)
        D_img = image_embedding.shape[1]
        D_text = text_embeddings_torch.shape[1]
        if D_img != D_text:
            # attempt automatic projeksiyon: varsa model.clip.visual_projection uygula,
            # yoksa model.clip.text_projection'in tersini kullanarak textleri projekte etmeye çalışmayız (tehlikeli),
            # onun yerine uyarı ver ve hata fırlat.
            applied = False
            if has_vis_proj:
                try:
                    image_embedding = _apply_projection(image_embedding, vis_proj)
                    applied = True
                except Exception as e:
                    print(f"[Warning] visual projection uygulanamadı: {e}")
            # ikinci şans: eğer text_proj varsa textleri ona göre değil de image boyutuna projekte edelim (daha güvenli değil ama deneyebiliriz)
            if not applied and has_text_proj:
                try:
                    # text_proj: (D_text_or_Din, D_embed) — dönüşümün tersini otomatik yapmak genelde mümkün değil.
                    # Bu yüzden text embeddingleri tekrar yükleyip model tarafında üretmek en güvenli çözüm.
                    raise RuntimeError(
                        "Görsel ve metin gömme boyutları farklı ve otomatik güvenli dönüşüm bulunamadı.\n"
                        "En güvenli çözüm: metin gömmelerini tekrar `save_text_embeddings` ile modelin text projeksiyonunu kullanarak kaydetmektir.\n"
                        "Veya modelde visual_projection parametresi ekleyip görsel gömmeyi aynı embed dimine projekte edin."
                    )
                except RuntimeError:
                    raise
            # son kontrol
            if image_embedding.shape[1] != text_embeddings_torch.shape[1]:
                raise RuntimeError(f"Görsel embed boyutu ({image_embedding.shape[1]}) ile kaydedilmiş metin embed boyutu ({text_embeddings_torch.shape[1]}) hala eşleşmiyor.")

        # sim: (B, N)
        sim = torch.matmul(image_embedding, text_embeddings_torch.T)

        topk_indices = torch.topk(sim, k=topk, dim=1).indices.cpu().numpy()

        import json

        with open("/content/CLIP4IDC/Second_CC_dataset/SECOND-CC-AUG/merged.json", "r") as f:
            mergedJson = json.load(f)   # data is now a dict

            for i, idx_list in enumerate(topk_indices):
                print(f"\n🖼 Image Pair {i}:")
                for rank, idx in enumerate(idx_list, start=1):
                    score = sim[i, idx].item()
                    sentence = mergedJson["images"][idx//5]["sentences"][idx-(idx//5)*5]["raw"]
                    print(str(rank)+" Sentence:" + sentence + " score: "+ str(score) + " id: " + str(idx))

    return topk_indices





def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    if args.n_gpu == 1:
        torch.distributed.init_process_group(
            backend='gloo',   # 'nccl' if using GPU, 'gloo' works for CPU as well
            init_method='tcp://127.0.0.1:29500',
            rank=0,
            world_size=1
        )

    tokenizer = ClipTokenizer()

    assert args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    # ####################################
    # freeze testing
    # ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():

            # top layers always need to train
            if (
                name.find("ln_final.") == 0
                or name.find("text_projection") == 0
                or name.find("logit_scale") == 0
                or name.find("visual.ln_post.") == 0
                or name.find("semantic_v.ln_post.") == 0
                or name.find("visual.proj") == 0
                or name.find("semantic_v.proj") == 0
                or name.find("visual.bef_embedding") == 0
                or name.find("semantic_v.bef_embedding") == 0
                or name.find("visual.aft_embedding") == 0
                or name.find("semantic_v.aft_embedding") == 0
                or name.find("visual.joint_positional_embedding") == 0
                or name.find("semantic_v.joint_positional_embedding") == 0
                or name.find("visual.ln_mid") == 0
                or name.find("semantic_v.ln_mid") == 0
                or name.find("clip.visual_fusion.fusion_layer") == 0
            ):
                continue  # need to train
            elif (
                name.find("visual.transformer.resblocks.") == 0
                or name.find("semantic_v.transformer.resblocks.") == 0
                or name.find("transformer.resblocks.") == 0
            ):
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue  # need to train

            if args.linear_patch == "3d" and name.find("conv2.") != -1:
                continue
            else:
                # parameters which < freeze_layer_num will be frozen
                param.requires_grad = False
                logger.info(f"Freeze layer: {name}")

    # ####################################
    # dataloader loading
    # ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](
            args,
            tokenizer,
            subset="val",
        )
    else:
        val_dataloader, val_length = test_dataloader, test_length

    # report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    logger.info("***** Running test *****")
    logger.info("  Test Num examples = %d", test_length)
    logger.info("  Test Batch size = %d", args.batch_size_val)
    logger.info("  Test Num steps = %d", len(test_dataloader))
    logger.info("***** Running val *****")
    logger.info("  Test Num examples = %d", val_length)

    # ####################################
    # train and eval
    # ####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (
            int(len(train_dataloader) + args.gradient_accumulation_steps - 1) / args.gradient_accumulation_steps
        ) * args.epochs

        # logger.info("*" * 80)
        # logger.info(enumerate(train_dataloader))
        # logger.info("*" * 80)

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)


        logger.info("***** Running training *****")
        logger.info("  Training Num examples = %d", train_length)
        logger.info("  Training Batch size = %d", args.batch_size)
        logger.info(
            "  Train Num steps = %d",
            num_train_optimization_steps * args.gradient_accumulation_steps,
        )

        best_score = 0.00001
        best_output_model_file = "None"
        # ##############################################################
        # resume optimizer state besides loss to continue train
        # ##############################################################

        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location="cpu")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            resumed_epoch = checkpoint["epoch"] + 1
            resumed_loss = checkpoint["loss"]  # noqa: F841

        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)  # Removed for single GPU
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, args.local_rank)
 

            logger.info(
                "Epoch %d/%s Finished, Train Loss: %f",
                epoch + 1,
                args.epochs,
                tr_loss,
            )

            output_model_file = save_model(
                epoch,
                args,
                model,
                optimizer,
                tr_loss,
                type_name="",
            )

            # Run on val dataset, this process is *TIME-consuming*.
            R1 = eval_epoch(args, model, test_dataloader, device)
            if best_score <= R1:
                best_score = R1
                best_output_model_file = output_model_file
            logger.info(
                f"The best model is: {best_output_model_file}, " f"the R1 is: {best_score:.4f}",
            )

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, device)
        #accumulate_vector()

    elif args.do_retrieval:
        # --- KULLANIM ---

        # Dosya yollarını kendine göre düzenle
        okuyucu = VeriSetiOkuyucu(
            tensor_path='tum_veri_seti_birlestirilmis.pt', 
            json_path='/content/CLIP4IDC/Second_CC_dataset/SECOND-CC-AUG/merged.json'
        )
        # ÖRNEK 1: 50. sıradaki veriyi çekelim
        veri_50 = okuyucu.get_item(50)
        print("\n--- 50. Kayıt Bilgisi ---")
        print(f"Resim Adı: {veri_50['image_file']}")
        print(f"Cümle:     {veri_50['text']}")
        print(f"sequence_output: {veri_50['sequence_output'].shape}")

        print(_run_on_single_gpu_retrieval(model, 50))



if __name__ == "__main__":
    main()