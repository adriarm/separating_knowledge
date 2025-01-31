#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

eval "$(conda shell.bash hook)"

EXPERIMENT_NAME='imagenet_base'
PRETRAINING_DATA_PATH='../data/imagenet/train'
KNN_GPU=4

./scripts/000_train_and_eval_default_dino_base.sh $EXPERIMENT_NAME $PRETRAINING_DATA_PATH $KNN_GPU