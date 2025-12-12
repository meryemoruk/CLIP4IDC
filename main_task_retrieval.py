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
    parser.add_argument("--do_save_vector", action="store_true")

    parser.add_argument("--dataloader_type", type=str, default="test")

    parser.add_argument("--index_retrieval", type=int, default=20)


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
    if not args.do_train and not args.do_eval and not args.do_retrieval and not args.do_save_vector:
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
    args,
    model,
    index,
    split = "test"
):
    # Dosya yollarını kendine göre düzenle
    if(args.dataloader_type == "train1"):
        split = "train"
    okuyucu = VeriSetiOkuyucu(
        tensor_path='tum_veri_seti_birlestirilmis.pt', 
        json_path='/content/CLIP4IDC/Second_CC_dataset/SECOND-CC-AUG/merged.json',
        split=split
    )

    json_path_catag = '/content/CLIP4IDC/Second_CC_dataset/SECOND-CC-AUG/merged_catag.json'

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

    result_np = np.array(result)

    # 2. flatten() ile tüm iç içe parantezleri kaldırıp tek bir düz çizgi haline getiriyoruz
    # Yeni boyut: (6121,) olacak. Artık dümdüz bir sayı dizisi.
    result_flat = result_np.flatten()

    # 3. Şimdi Tensor'a çevirip Top-K işlemini yapabiliriz
    result_tensor = torch.from_numpy(result_flat)
    # 1. Adım: argsort ile sıralama yapıldığında elemanların nereye gideceğini (indeksleri) bulur.
    # Bu küçükten büyüğe sıralar, o yüzden sonuna dilimleme ekleriz.
    top_5_degerler, top_5_indeksler = torch.topk(result_tensor, k=5)
    tum_sirali_indeksler = torch.argsort(result_tensor, descending=True)

    # Eğer bunları ekrana yazdırmak veya listeye çevirmek istersen:
    top_5_deger_listesi = top_5_degerler.cpu().numpy().tolist()
    top_5_indeks_listesi = top_5_indeksler.cpu().numpy().tolist()

    top_5_captions = []
    top_5_images = []
    for i in top_5_indeks_listesi:
        tempData = okuyucu.get_item(i)
        top_5_captions.append(tempData["text"])
        top_5_images.append(os.path.join("/content/CLIP4IDC/Second_CC_dataset/SECOND-CC-AUG", split, "rgb", "A",
            tempData["image_file"]))
        top_5_images.append(os.path.join("/content/CLIP4IDC/Second_CC_dataset/SECOND-CC-AUG", split, "rgb", "B",
            tempData["image_file"]))
        
    
    import shutil

    # 1. Create the folder if it doesn't exist
    target_folder = 'retrivalImages/'
    os.makedirs(target_folder, exist_ok=True)
    
    for i, path in enumerate(top_5_images):
        image_name = str(i) + ".png"
        dst_path = os.path.join(target_folder, image_name)
        # copy2 preserves metadata (creation time, modification time, etc.)
        shutil.copy2(path, dst_path)

        print(f"Copied to {dst_path}")
    

    print(f"Original Resim Benzerliği: {result_flat[index]}")
    print(f"Top 5 Benzerlikler: {top_5_deger_listesi}")
    print(f"Top 5 İndeksler: {top_5_indeks_listesi}")
    print(f"Top 5 Captionlar: {top_5_captions}")
    print(f"Top 5 Images: {top_5_images}")

    og_index = -1
    for i, m in enumerate(tum_sirali_indeksler):
        if(m == index):
            og_index = i
            break
    
    data_found = []
    found_name = []
    catag_id = [-1,-1,-1]
    og_catag_id = -1
    
    import json
    with open(json_path_catag, 'r') as f:
        # Parse file content directly into a Python object
        jsonCatag = json.load(f)
        for image_entry in jsonCatag['images']:
                if image_entry.get('filename') == data["image_file"]:
                    og_catag_id = image_entry["category"]
                    break
        for i in range(0,3):
            data_found.append(okuyucu.get_item(tum_sirali_indeksler[i])) 
            found_name.append(data_found[i]["image_file"])
            for image_entry in jsonCatag['images']:
                if image_entry.get('filename') == found_name[i]:
                    catag_id[i] = image_entry["category"]
                    break


    import json

    inference_result = {
        "index": index,
        "rank": og_index,
        "confidence": top_5_deger_listesi[0],
        "o_catag": og_catag_id,
        "f_catag_1": catag_id[0],
        "f_catag_2": catag_id[1],
        "f_catag_3": catag_id[2]
    }

    

    with open("inference_results.json", "a", encoding="utf-8") as f:
        json.dump(inference_result, f, indent=4) # indent=4 okunabilir formatlar



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
    
    R1 = tv_metrics["R1"]
    return R1

def eval_epoch_save(args, model, test_dataloader, device):
    """
    Modelden sequence ve visual outputları çıkarır ve bir .pt dosyasına kaydeder.
    """
    if hasattr(model, "module"):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    all_sequence_outputs = []
    all_visual_outputs = []
    all_input_masks = []
    all_pair_masks = [] # Görüntü maskeleri
    all_image_filenames = []
    all_texts = []
    all_segment_ids = []

    logger.info("Veri seti vektörleri çıkarılıyor ve birleştiriliyor...")

    with torch.no_grad():
        for bid, batch in enumerate(test_dataloader):
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

            # 1. Metin Feature Çıkarma
            sequence_output, _ = model.get_sequence_output(
                input_ids,
                segment_ids,
                input_mask,
            )

            # 2. Görüntü Feature Çıkarma
            # CLIP4IDC yapısına göre image pair birleştiriliyor
            image_pair = torch.cat([bef_image, aft_image], 1)
            semantic_pair = torch.cat([bef_semantic, aft_semantic], 1)
            
            visual_output, _ = model.get_visual_output(
                image_pair,
                semantic_pair,
                image_mask,
            )

            # 3. CPU'ya alıp listeye ekle (GPU hafızası şişmesin diye)
            all_sequence_outputs.append(sequence_output.cpu())
            all_visual_outputs.append(visual_output.cpu())
            all_input_masks.append(input_mask.cpu())
            all_pair_masks.append(image_mask.cpu())
            all_segment_ids.append(segment_ids.cpu())

            # 4. Dosya isimlerini ve metinleri dataloader'dan almak gerekebilir
            # Eğer dataloader batch içinde raw text veya filename döndürmüyorsa,
            # dataset objesinden indeksle erişmemiz gerekebilir.
            # Burada toplu işlem (batch) indekslerini takip ederek datasetten çekiyoruz:
            
            # Not: Distributed sampler kullanılıyorsa indeksler karışık olabilir, 
            # ancak test modunda genelde sequential sampler kullanılır.
            start_idx = bid * args.batch_size_val
            end_idx = start_idx + input_ids.size(0)
            
            for i in range(start_idx, end_idx):
                if i < len(test_dataloader.dataset):
                    # Dataset yapınıza göre bu key'ler değişebilir ('image_file', 'text')
                    # CLIP4IDC dataset yapısında genelde __getitem__ dict döner ama 
                    # dataloader collate_fn ile tensor yapar. Raw dataya dataset.raw_data veya benzeri bir yerden ulaşırız.
                    # Eğer datasetinizde doğrudan get_item varsa:
                    # (Bu kısım dataset sınıfınızın yapısına bağlı, varsayımsal yazıyorum)
                    try:
                        raw_item = test_dataloader.dataset.get_raw_item(i) # Özel bir fonksiyon varsayıyoruz
                        all_image_filenames.append(raw_item['image_file'])
                        all_texts.append(raw_item['text'])
                    except:
                        # Fallback: Eğer get_raw_item yoksa, veri setini tekrar okumak yerine
                        # bu adımı sonraya bırakabiliriz veya datasetin içindeki listeden alabiliriz.
                        # Örnek:
                        if hasattr(test_dataloader.dataset, 'data_list'):
                             item = test_dataloader.dataset.data_list[i]
                             all_image_filenames.append(item.get('image_file', 'unknown.jpg'))
                             all_texts.append(item.get('text', ''))
                        else:
                            all_image_filenames.append("unknown.jpg")
                            all_texts.append("unknown text")

            if bid % 10 == 0:
                logger.info(f"Batch {bid}/{len(test_dataloader)} işlendi.")

    # Tüm listeleri tek bir büyük tensora/listeye çevir
    save_data = {
        "sequence_output": torch.cat(all_sequence_outputs, dim=0),
        "visual_output": torch.cat(all_visual_outputs, dim=0),
        "input_mask": torch.cat(all_input_masks, dim=0),
        "pair_mask": torch.cat(all_pair_masks, dim=0),
        "segment_ids": torch.cat(all_segment_ids, dim=0),
        "image_filenames": all_image_filenames,
        "texts": all_texts
    }

    output_path = os.path.join(args.output_dir, "tum_veri_seti_birlestirilmis.pt")
    torch.save(save_data, output_path)
    logger.info(f"✅ Tüm vektörler kaydedildi: {output_path}")
    logger.info(f"Toplam Veri Sayısı: {len(all_image_filenames)}")

def run_retrieval_pipeline(args, model, device):
    import json
    import os  # <--- DÜZELTME: os modülü gerekli
    from tqdm import tqdm

    # 1. Kaydedilmiş Vektörleri Yükle
    pt_path = os.path.join(args.output_dir, "tum_veri_seti_birlestirilmis.pt")
    if not os.path.exists(pt_path):
        logger.error("PT dosyası bulunamadı! Önce --do_save_vector çalıştırın.")
        return

    logger.info(f"Vektörler yükleniyor: {pt_path}")
    data = torch.load(pt_path, map_location=device, weights_only=True)
    
    sequence_outputs = data["sequence_output"]
    visual_outputs = data["visual_output"]
    input_masks = data["input_mask"]
    pair_masks = data["pair_mask"]
    image_filenames = data["image_filenames"]

    num_samples = sequence_outputs.size(0)
    logger.info(f"Toplam örnek sayısı: {num_samples}")

    # 2. Kategori Verisini Yükle
    json_path_catag = args.json_path 
    logger.info(f"Kategori dosyası yükleniyor: {json_path_catag}")
    
    image_to_category = {}
    with open(json_path_catag, 'r') as f:
        json_catag = json.load(f)
        for img_entry in json_catag.get('images', []):
            # JSON'daki filename'in de temiz olduğundan emin olalım
            clean_json_name = os.path.basename(img_entry['filename']) 
            image_to_category[clean_json_name] = img_entry['category']

    # DEBUG: Hatanın sebebini görmek için ilk kaydı ekrana basıyoruz
    if len(image_filenames) > 0:
        ornek_pt_ismi = image_filenames[0]
        ornek_pt_ismi_temiz = os.path.basename(ornek_pt_ismi)
        logger.info(f"DEBUG KONTROL:")
        logger.info(f"PT dosyasındaki ham isim: '{ornek_pt_ismi}'")
        logger.info(f"Temizlenmiş isim: '{ornek_pt_ismi_temiz}'")
        logger.info(f"JSON'da bu isim var mı?: {ornek_pt_ismi_temiz in image_to_category}")
        if ornek_pt_ismi_temiz in image_to_category:
            logger.info(f"Kategorisi: {image_to_category[ornek_pt_ismi_temiz]}")
        else:
            logger.info("JSON anahtarlarından ilk 5 tanesi:")
            logger.info(list(image_to_category.keys())[:5])

    # 3. Hesaplama ve Kaydetme Döngüsü
    inference_results = []
    
    if hasattr(model, "module"):
        model = model.module
    model.eval()
    model.to(device)

    results_file = "inference_results_all.json"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("[\n") 

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Retrieval"):
            
            query_seq = sequence_outputs[i].unsqueeze(0).to(device)
            query_mask = input_masks[i].unsqueeze(0).to(device)
            
            all_logits = []
            batch_size_sim = 128 
            
            for j in range(0, num_samples, batch_size_sim):
                end_j = min(j + batch_size_sim, num_samples)
                vis_batch = visual_outputs[j:end_j].to(device)
                mask_batch = pair_masks[j:end_j].to(device)
                
                logits, *_ = model.get_similarity_logits(
                    query_seq, 
                    vis_batch, 
                    query_mask, 
                    mask_batch
                )
                all_logits.append(logits.cpu()) 
            
            full_logits = torch.cat(all_logits, dim=-1).flatten() 
            
            top_k_scores, top_k_indices = torch.topk(full_logits, k=3)
            
            top_indices = top_k_indices.numpy().tolist()
            top_scores = top_k_scores.numpy().tolist()
            
            # --- DÜZELTME BURADA BAŞLIYOR ---
            
            # 1. Sorgu resminin ismini temizle (Path'i at)
            query_img_raw = image_filenames[i]
            query_img_clean = os.path.basename(query_img_raw)
            
            og_category = image_to_category.get(query_img_clean, -1)
            
            found_categories = []
            for idx in top_indices:
                # 2. Bulunan resimlerin isimlerini temizle
                found_img_raw = image_filenames[idx]
                found_img_clean = os.path.basename(found_img_raw)
                
                cat_id = image_to_category.get(found_img_clean, -1)
                found_categories.append(cat_id)

            # --- DÜZELTME BURADA BİTİYOR ---
            
            result_entry = {
                "index": i,
                "query_image": query_img_clean, # Temiz ismi kaydetmek daha okunaklı olur
                "rank": -1, 
                "confidence": top_scores[0], 
                "o_catag": og_category,
                "f_catag_1": found_categories[0],
                "f_catag_2": found_categories[1],
                "f_catag_3": found_categories[2]
            }
            
            with open(results_file, "a", encoding="utf-8") as f:
                json.dump(result_entry, f, indent=4)
                if i < num_samples - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
        
    with open(results_file, "a", encoding="utf-8") as f:
        f.write("]") 

    logger.info(f"✅ Çıkarım tamamlandı. Sonuçlar: {results_file}")


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

    assert DATALOADER_DICT[args.datatype][args.dataloader_type] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype][args.dataloader_type] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype][args.dataloader_type](args, tokenizer)

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
        train_eval_dataloader, train_eval_length = DATALOADER_DICT[args.datatype]["train1"](
            args,
            tokenizer,
            subset="train1",
        )
        
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
            logger.info("Start Loading Optimizer.")
            checkpoint_opt = torch.load(args.resume_model_opt, map_location='cpu')
            resumed_epoch = checkpoint_opt['epoch']+1
            optimizer.load_state_dict(checkpoint_opt['optimizer_state_dict'])
            resumed_loss = checkpoint_opt['loss']
            logger.info("End Loading Optimizer.")
            if 'optimizer_state_dict' in checkpoint_opt:
                ckpt_groups = len(checkpoint_opt['optimizer_state_dict']['param_groups'])
                curr_groups = len(optimizer.param_groups)
                logger.info(f"Optimizer param_groups -> checkpoint: {ckpt_groups}, current: {curr_groups}")

        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                if(epoch%10 == 0):
                    logger.info("--------------------Eval on train dataset------------------")
                    R1 = eval_epoch(args, model, train_eval_dataloader, device)

                ## Run on val dataset, this process is *TIME-consuming*.
                logger.info("--------------------Eval on val dataset------------------")
                R1 = eval_epoch(args, model, val_dataloader, device)

                #R1 = eval_epoch(args, model, test_dataloader, device)
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, device)

    elif args.do_save_vector:
        # Hangi dataloader kullanılacaksa
        if args.dataloader_type == "train1":
             subset_name = "train1"
             dl_func = DATALOADER_DICT[args.datatype]["train1"]
        else:
             subset_name = args.dataloader_type
             dl_func = DATALOADER_DICT[args.datatype][args.dataloader_type]

        if dl_func is not None:
             dataloader, length = dl_func(args, tokenizer, subset=subset_name if subset_name=="train1" else "test")
             logger.info(f"Dataloader yüklendi: {length} örnek.")
             
             # 1. Adım: Kaydetme
             eval_epoch_save(args, model, dataloader, device)
        else:
            logger.error("Dataloader bulunamadı!")

    elif args.do_retrieval:
        # --- KULLANIM ---

        run_retrieval_pipeline(args, model, device)
            





if __name__ == "__main__":
    main()
