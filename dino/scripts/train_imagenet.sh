#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

eval "$(conda shell.bash hook)"

EXPERIMENT_NAME='imagenet'
PRETRAINING_DATA_PATH='../data/imagenet/train'
KNN_GPU=4

./scripts/000_train_and_eval_default_dino.sh $EXPERIMENT_NAME $PRETRAINING_DATA_PATH $KNN_GPU