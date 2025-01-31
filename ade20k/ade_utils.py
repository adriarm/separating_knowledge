import sys
# sys.path.append('.')
# sys.path.append('../..')
# sys.path.append('my_python_utils')

import os
import glob
# import seaborn as sns

# from my_python_utils.common_utils import *
# from ssl_utils import *

from scipy import stats
import torch
import torchvision

import json

import PIL
from PIL import Image

import numpy as np

# from https://github.com/CSAILVision/ADE20K/blob/main/utils/utils_ade20k.py
def loadAde20K(file):
    assert file.endswith('.jpg')
    fileseg = file.replace('.jpg', '_seg.png')
    with Image.open(fileseg) as io:
        seg = np.array(io)

    # Img
    # img = torchvision.io.read_image(file)
    img = PIL.Image.open(file)

    # Obtain the segmentation mask, bult from the RGB channels of the _seg file
    R = seg[:,:,0]
    G = seg[:,:,1]
    B = seg[:,:,2]
    ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32))


    # Obtain the instance mask from the blue channel of the _seg file
    Minstances_hat = np.unique(B, return_inverse=True)[1]
    Minstances_hat = np.reshape(Minstances_hat, B.shape)
    ObjectInstanceMasks = Minstances_hat


    level = 0
    PartsClassMasks = []
    PartsInstanceMasks = []
    while True:
        level = level+1
        file_parts = file.replace('.jpg', '_parts_{}.png'.format(level))
        if os.path.isfile(file_parts):
            with Image.open(file_parts) as io:
                partsseg = np.array(io)
            R = partsseg[:,:,0]
            G = partsseg[:,:,1]
            B = partsseg[:,:,2]
            PartsClassMasks.append((np.int32(R)/10)*256+np.int32(G))
            PartsInstanceMasks = PartsClassMasks
            # TODO:  correct partinstancemasks

            
        else:
            break

    objects = {}
    parts = {}

    attr_file_name = file.replace('.jpg', '.json')
    if os.path.isfile(attr_file_name):
        with open(attr_file_name, 'r') as f:
            input_info = json.load(f)

        contents = input_info['annotation']['object']
        instance = np.array([int(x['id']) for x in contents])
        names = [x['raw_name'] for x in contents]
        corrected_raw_name =  [x['name'] for x in contents]
        partlevel = np.array([int(x['parts']['part_level']) for x in contents])
        ispart = np.array([p>0 for p in partlevel])
        iscrop = np.array([int(x['crop']) for x in contents])
        listattributes = [x['attributes'] for x in contents]
        polygon = [x['polygon'] for x in contents]
        for p in polygon:
            p['x'] = np.array(p['x'])
            p['y'] = np.array(p['y'])

        objects['instancendx'] = instance[ispart == 0]
        objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
        objects['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 0)[0])]
        objects['iscrop'] = iscrop[ispart == 0]
        objects['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 0)[0])]
        objects['polygon'] = [polygon[x] for x in list(np.where(ispart == 0)[0])]


        parts['instancendx'] = instance[ispart == 1]
        parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
        parts['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 1)[0])]
        parts['iscrop'] = iscrop[ispart == 1]
        parts['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 1)[0])]
        parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

    return {'img_name': file, 'segm_name': fileseg,
            'img':img,
            'class_mask': ObjectClassMasks, 'instance_mask': ObjectInstanceMasks, 
            'partclass_mask': PartsClassMasks, 'part_instance_mask': PartsInstanceMasks, 
            'objects': objects, 'parts': parts}

class ADEDataset(torch.utils.data.Dataset):

  def __init__(self, ade_paths='ade20k/paths.json', transform=None) -> None:
    super().__init__()
    
    # Calculate paths
    # paths = glob.glob("/data/vision/torralba/datasets/ade20k/ADE20K_2021_17_01/images/ADE/training/**/*.jpg", recursive=True)
    # Save paths
    # with open('ade20k/paths.json', 'w') as f:
    #   json.dump(paths, f)
    # Load paths
    with open(ade_paths) as f:
      self.paths = json.load(f)

    self.transform = transform

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    img = loadAde20K(self.paths[index])
    if self.transform is not None:
        return self.transform(img)
    else:
        return img