import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import timm
import torch
import torchvision
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torchvision.datasets.folder import pil_loader
from tqdm.auto import tqdm


class MixupDataset(torch.utils.data.Dataset):
  def __init__(self, ds, N):
    self.ds = ds
    self.N = N
  
  def __len__(self):
    return len(self.ds)

  def __getitem__(self, index):
    rnggen = np.random.default_rng(seed=index)
    samples = [index] + list(rnggen.choice(len(self.ds), self.N-1))
    images = [self.ds[i][0] for i in samples]
    mixup_image = torch.stack(images).mean(0)

    return mixup_image, index

def save_dataset(ds, output_folder):
  for i in tqdm(range(len(ds))):
    img = ds[i][0]
    base_folder_path = output_folder + '/base_' + str((i // 1000) * 1000).zfill(10)
    os.makedirs(base_folder_path, exist_ok=True)
    base_image_path = base_folder_path + '/' + str(i).zfill(10) + '.jpg'
    torchvision.utils.save_image(img, base_image_path)

def generate_mixup_dataset(input_folder, output_folder, N):
  source_ds = torchvision.datasets.ImageFolder(input_folder, transform=torchvision.transforms.ToTensor())
  target_ds = MixupDataset(source_ds, N)
  save_dataset(target_ds, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('input_folder', type=str, help='input folder')
    parser.add_argument('output_folder', type=str, help='output folder')
    
    parser.add_argument('N', type=int, help='Number of samples')

    args = parser.parse_args()

    generate_mixup_dataset(args.input_folder, args.output_folder, args.N)