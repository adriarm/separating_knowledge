import os
import sys
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import torchvision.transforms as T  # Optional for transformations

import utils
import vision_transformer as vits

from pathlib import Path

class COCOSegmentationDataset(Dataset):
    def __init__(self, images_dir, annotations_path, res=448, transform=None):
        """
        Args:
            images_dir (str): Path to the images directory.
            annotations_path (str): Path to the COCO annotations file.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.images_dir = images_dir
        self.coco = COCO(annotations_path)
        self.image_ids = self.coco.getImgIds()
        self.res = res
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Load image
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.res, self.res))

        # Load mask
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        
        # Initialize an empty mask
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        
        # Populate the mask with annotations
        for annotation in annotations:
            category_id = annotation['category_id']
            binary_mask = self.coco.annToMask(annotation)
            mask[binary_mask == 1] = category_id
        mask = cv2.resize(mask, (self.res // 16, self.res // 16), interpolation=cv2.INTER_NEAREST)

        # Apply transforms if any
        if self.transform:
            image_tensor = self.transform(image_rgb)
        else:
            image_tensor = torch.tensor(image_rgb)

        # Convert to PyTorch tensors
        image_tensor = image_tensor #.permute(2, 0, 1)  # CxHxW format
        mask = torch.tensor(mask, dtype=torch.long)  # Mask in HxW format

        return image_rgb, image_tensor, mask