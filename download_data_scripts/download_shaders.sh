#!/bin/bash
mkdir -p data

datasets=('shaders21k' \
          'shaders21k_6mixup'
           )

for DATASET in ${datasets[@]}
do
    echo "Downloading $DATASET"
    wget -O $1/$DATASET.zip http://data.csail.mit.edu/synthetic_training/shaders21k/zipped_data/$DATASET.zip
    yes | unzip $1/$DATASET.zip -d $1/$DATASET
    rm $1/$DATASET.zip
done