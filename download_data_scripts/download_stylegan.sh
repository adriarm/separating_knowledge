#!/bin/bash

$DATASET=stylegan-oriented
echo "Downloading $DATASET"
wget -O $1/$DATASET.zip http://data.csail.mit.edu/noiselearning/zipped_data/large_scale/$DATASET.zip
yes | unzip $1/$DATASET.zip -d $1/$DATASET
rm $2/$DATASET.zip