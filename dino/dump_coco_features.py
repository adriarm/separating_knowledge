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

def main(args):
  from torchvision import transforms

  # Define the transforms
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  dataset_train = COCOSegmentationDataset(images_dir = "../data/coco2017/train2017",
                                    annotations_path = "../data/coco2017/annotations/instances_train2017.json",
                                    transform=transform)

  dataset_val = COCOSegmentationDataset(images_dir = "../data/coco2017/val2017",
                                    annotations_path = "../data/coco2017/annotations/instances_val2017.json",
                                    transform=transform)
  
  data_loader_train = DataLoader(
    dataset_train,
    batch_size=args.batch_size_per_gpu,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
    shuffle=False
  )
  data_loader_val = DataLoader(
    dataset_val,
    batch_size=args.batch_size_per_gpu,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
    shuffle=False
  )
  
  # ============ building network ... ============
  model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
  print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
  model.cuda()
  utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size, pretraining_epochs=args.pretraining_epochs)
  model.eval()

  # ============ create feature save folder ============
  resolution = dataset_train[0][0].shape[1] // args.patch_size
  dump_features = str(Path(args.pretrained_weights).parent) + f'/checkpoint/coco_dumps/coco_dump_{resolution}'
  os.makedirs(dump_features, exist_ok=True)

  # ============ extract and save train features ============
  print("Extracting features for train set...")
  # train_images_rgb, train_features, train_masks = extract_features(model, data_loader_train)
  # extract_features(model, data_loader_train, dump_features, 'train')
  #torch.save(train_features, os.path.join(dump_features, "trainfeat.pth"))
  #del train_features

  # ============ extract and save val features ============
  print("Extracting features for val set...")
  # test_images_rgb, test_features, test_masks = extract_features(model, data_loader_val)
  extract_features(model, data_loader_val, dump_features, 'val')
  #torch.save(test_features, os.path.join(dump_features, "testfeat.pth"))
  #del test_features

  # # ============ save images and masks ==================
#   dump_images_and_masks = '../data/coco_dumps'
#   if not os.path.isfile(os.path.join(dump_images_and_masks, f'trainimages_{resolution}.pth')):
#     torch.save(train_images_rgb, os.path.join(dump_images_and_masks, f"trainimages_{resolution}.pth"))
#     torch.save(test_images_rgb, os.path.join(dump_images_and_masks, f"testimages_{resolution}.pth"))
#     torch.save(train_masks, os.path.join(dump_images_and_masks, f"trainmasks_{resolution}.pth"))
#     torch.save(test_masks, os.path.join(dump_images_and_masks, f"testmasks_{resolution}.pth"))

@torch.no_grad()
def extract_features(model, data_loader, dump_features, split):
    metric_logger = utils.MetricLogger(delimiter="  ")
    batch_idx = 0
    for image_rgb, image_tensor, mask in metric_logger.log_every(data_loader, 10):
      image_tensor = image_tensor.cuda(non_blocking=True)
      with torch.inference_mode():
        feats = model.get_intermediate_layers(image_tensor)[0].squeeze()
      torch.save(feats, os.path.join(dump_features, f"{split}feat_{batch_idx}.pth"))
      batch_idx += 1

class COCOSegmentationDataset(Dataset):
    def __init__(self, images_dir, annotations_path, transform=None):
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
        image_rgb = cv2.resize(image_rgb, (448, 448))

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
        mask = cv2.resize(mask, (448 // 16, 448 // 16), interpolation=cv2.INTER_NEAREST)

        # Apply transforms if any
        if self.transform:
            image_tensor = self.transform(image_rgb)
        else:
            image_tensor = torch.tensor(image_rgb)

        # Convert to PyTorch tensors
        image_tensor = image_tensor #.permute(2, 0, 1)  # CxHxW format
        mask = torch.tensor(mask, dtype=torch.long)  # Mask in HxW format

        return image_rgb, image_tensor, mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')

    # our extra arguments
    parser.add_argument('--pretraining-epochs', default=100, type=int, help='Number of epochs of pretraining. Used to check that the epoch is correctly loaded')
    args = parser.parse_args()

    main(args)
