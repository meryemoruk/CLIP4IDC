import torch as th
import numpy as np
from PIL import Image

# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoProcessor

# pip install opencv-python
import cv2


class RawImageExtractorCV2:
    def __init__(self, centercrop=False, size=224):
        self.centercrop = centercrop
        self.size = size
        self.transform = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    def image_to_tensor(self, image_file, preprocess):
        image = Image.open(image_file).convert("RGB")
        image_data = preprocess(images=image, return_tensors="pt")

        return {"image": image_data}

    def get_image_data(self, image_path):
        image_input = self.image_to_tensor(image_path, self.transform)
        return image_input


# An ordinary video frame extractor based CV2
RawImageExtractor = RawImageExtractorCV2
