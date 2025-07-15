# Separating Knowledge and Perception with Procedural Data

Welcome to the repository for "Separating Knowledge with Procedural Data"! Here are the instructions for reproducing all trainings and evaluations in the paper.

<p align="center">
  <img width="100%" src="https://adriarm.github.io/_pages/separating_knowledge/static/figures_poster/diagram_small.png">
</p>

[[Project page](https://adriarm.github.io/_pages/separating_knowledge/)] 
[[Paper](https://openreview.net/pdf?id=oyFAFpqaca)]


# Creating environment
```
conda create -n separating_knowledge python=3.9.20
conda activate separating_knowledge
pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1
pip install importlib-metadata
pip install opencv-python
pip install timm
pip install matplotlib
pip install pandas
pip install pykeops
pip install seaborn
pip install medmnist
pip install pycocotools
pip install scikit-learn
```

# Downloading and generating data
Follow each subsection instructions, starting each subsection in the main directory.

## Downloading ImageNet, Places
See their respective websites for instructions [ImageNet](https://image-net.org/download.php) and [Places](http://places.csail.mit.edu), then symlink to `./data`.

## Downloading Stylegan, Shaders, Shaders Mixup
```
cd download_data_scripts
./download_stylegan.sh YOUR_DATASETS_FOLDER_HERE
./download_shaders.sh YOUR_DATASETS_FOLDER_HERE
```
## Setting up data symlinks for training scripts
```
mkdir data
cd data

mkdir imagenet
mkdir places
mkdir shaders_mixup
mkdir shaders
mkdir stylegan

ln -s PATH_TO_IMAGENET/train imagenet/train
ln -s PATH_TO_IMAGENET/val imagenet/val
ln -s PATH_TO_PLACES/train places/train
ln -s PATH_TO_SHADERS_MIXUP/train shaders_mixup/train
ln -s PATH_TO_SHADERS/train shaders/train
ln -s PATH_TO_STYLEGAN/train stylegan/train
```

## Creating Shaders KML and Shaders KML Mixup
```
cd data_generation

# Shaders KML
./shaders_kml.sh PATH_TO_DATASET_FOLDER
mkdir ../data/shaders_kml
ln -s PATH_TO_SHADERS_KML/train ../data/shaders_kml/train

# Shaders KML Mixup
./shaders_kml_mixup.sh PATH_TO_DATASET_FOLDER
mkdir ../data/shaders_kml_mixup
ln -s PATH_TO_SHADERS_KML_MIXUP/train ../data/shaders_kml_mixup/train
```

## Download CUB200, Flowers102, Food101
See their respective websites for instructions [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/), [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), and [Food](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), then symlink to `./data`.

## Download MedicalMNIST
See the website [MedicalMNIST](https://medmnist.com) for instructions, download in 224x244 resolution and then symlink to `./data`.

## Download COCO, Ade20k, Pascal-VOC
See their respective websites for instructions [COCO](https://cocodataset.org/#download), [Ade20k](https://ade20k.csail.mit.edu), and [Pascal](http://host.robots.ox.ac.uk/pascal/VOC/), then symlink to `./data`.

# Training + ImageNet-1K evaluation
```
cd dino
./scripts/train_imagenet.sh
./scripts/train_places.sh
./scripts/train_shaders_kml_mixup.sh
./scripts/train_shaders_kml.sh
./scripts/train_shaders_mixup.sh
./scripts/train_shaders.sh
./scripts/train_stylegan.sh
```

# Evaluation

## Fine-grained dataset evaluation
```
cd dino

# ENCODER_NAME=imagenet, shaders_kml_mixup, etc.
./scripts/evals/eval_knn_cub.sh ENCODER_NAME # ENCODER_NAME=imagenet, shaders_kml_mixup, etc.
./scripts/evals/eval_knn_flowers.sh ENCODER_NAME # ENCODER_NAME=imagenet, shaders_kml_mixup, etc.
./scripts/evals/eval_knn_food.sh ENCODER_NAME # ENCODER_NAME=imagenet, shaders_kml_mixup, etc.
```

## Medical evaluation
```
cd dino

# ENCODER_NAME=imagenet, shaders_kml_mixup, etc.
# DATASET_NAME=bloodmnist, breastmnist, dermamnist, octmnist, organamnist, organcmnist, organsmnist, pathmnist, pneumoniamnist, tissuemnist
./scripts/evals/eval_knn_medicalmnist.sh ENCODER_NAME DATASET_NAME 
```

## Segmentation evaluation
### Dump COCO image features
```
cd dino

# ENCODER_NAME=imagenet, shaders_kml_mixup, etc.
python dump_coco_features.py --pretrained_weights ./encoders/ENCODER_NAME/checkpoint.pth
```
### Go to notebooks
Open notebooks `notebook_figures/figures_segmentation_zeroshot.ipynb`, `notebook_figures/figures_segmentation_incontext.ipynb`, and `notebook_figures/figures_segmentation_knn.ipynb`.

