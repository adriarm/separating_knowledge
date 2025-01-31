import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import skimage
import timm
import torch
import torchvision
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torchvision.datasets.folder import pil_loader
from tqdm.auto import tqdm
import cv2

import utils_kmeans

class Fast2LeavesDataset(torch.utils.data.Dataset):
  def __init__(self, shapes_ds, textures_ds, scale=(1.0, 1.0), seed=0):
    self.shapes_ds = shapes_ds
    self.textures_ds = textures_ds
    self.K = 2
    # self.scale = scale
    self.seed = seed
  
  def __len__(self):
    return len(self.shapes_ds)
  
  def get_textures(self, x0, rnggen):
    texture_idxs = rnggen.choice(len(self.textures_ds), self.K)
    textures_samples = torch.stack([self.textures_ds[i][0] for i in texture_idxs])

    resize_fn = torchvision.transforms.Resize(x0.shape[1])
    textures_samples = resize_fn(textures_samples)

    return textures_samples
  
  def get_clusters(self, x, rnggen):
    x_shape = x.shape
    x = torchvision.transforms.Resize((256, 256))(x)
    tokens = x.view(3, -1).T.contiguous().numpy()
    # print(tokens.shape)
    # cl, c = utils_kmeans.KMeans(tokens, rnggen, self.K, Niter=3)
    cl, c = utils_kmeans.Fast2Means(tokens, rnggen, Niter=3)
    cl = (cl == 0).astype(np.float32)
    # print(c.shape)
    cl = cl.reshape(x.shape[1], x.shape[2])
    cl = cv2.blur(cl, (3, 3))
    cl = cv2.resize(cl, (x_shape[1], x_shape[2]), interpolation=cv2.INTER_LINEAR)
    cl = cv2.blur(cl, (3, 3))
    return cl

  def __getitem__(self, index):
    rnggen = np.random.default_rng(seed=len(self)*self.seed + index)

    x0, y0 = self.shapes_ds[index][:2]
    # x0 = torchvision.transforms.RandomResizedCrop(x0.shape[-2:], scale=self.scale, ratio=(0.75, 1.3333333333333333))(x0)
    x = torch.zeros_like(x0)
    textures_samples = self.get_textures(x0, rnggen)
    
    cl = self.get_clusters(x0, rnggen)
    x += textures_samples[0] * (cl)
    x += textures_samples[1] * (1 - cl)

    return x, y0 #, (x0, cl, textures_samples)
  
  # def display(self, seed=0, num_examples=8):
  #   rnggen = np.random.default_rng(seed=seed)
  #   to_pil = torchvision.transforms.ToPILImage()
  #   images_list = []
  #   for i in rnggen.choice(len(self), num_examples):
  #     x, y0, (x0, cl, textures_samples) = self[i]
  #     colorized_cl = 255 * cl[:, :, None] #soft_colorize_cluster_assignments(cl[:, :, None], self.K)
  #     images = [x, x0, colorized_cl, *textures_samples]
  #     images_list.append(hconcat_pil_images([to_pil(x) for x in images]))
  #   display(vconcat_pil_images(images_list))

def save_dataset(ds, output_folder):
  # Resume if cancelled
  num_generated = 0
  try:
    partial_ds = torchvision.datasets.ImageFolder(output_folder, transform=torchvision.transforms.ToTensor())
    num_generated = len(partial_ds)
  except:
    pass

  for i in tqdm(range(num_generated, len(ds))):
    img = ds[i][0]
    base_folder_path = output_folder + '/base_' + str((i // 1000) * 1000).zfill(10)
    os.makedirs(base_folder_path, exist_ok=True)
    base_image_path = base_folder_path + '/' + str(i).zfill(10) + '.jpg'
    torchvision.utils.save_image(img, base_image_path)

def generate_kleaves_dataset(shapes_folder, textures_folder, output_folder):
  shapes_ds = torchvision.datasets.ImageFolder(shapes_folder, transform=torchvision.transforms.ToTensor())
  textures_ds = torchvision.datasets.ImageFolder(textures_folder, transform=torchvision.transforms.ToTensor())
  target_ds = Fast2LeavesDataset(shapes_ds, textures_ds)
  save_dataset(target_ds, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('shapes_folder', type=str, help='shapes folder path')
    parser.add_argument('textures_folder', type=str, help='textures folder path')
    parser.add_argument('output_folder', type=str, help='output folder')

    args = parser.parse_args()

    generate_kleaves_dataset(args.shapes_folder, args.textures_folder, args.output_folder)