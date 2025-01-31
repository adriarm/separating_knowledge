#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

eval "$(conda shell.bash hook)"

EXPERIMENT_NAME='imagenet_tiny'
PRETRAINING_DATA_PATH='../data/imagenet/train'
KNN_GPU=4

./scripts/000_train_and_eval_default_dino_tiny.sh $EXPERIMENT_NAME $PRETRAINING_DATA_PATH $KNN_GPU