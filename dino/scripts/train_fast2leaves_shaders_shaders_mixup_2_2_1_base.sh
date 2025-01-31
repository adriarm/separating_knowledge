#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

eval "$(conda shell.bash hook)"

EXPERIMENT_NAME='shaders_kml_mixup_base'
PRETRAINING_DATA_PATH='../data/shaders_kml_mixup/train'
KNN_GPU=6

./scripts/000_train_and_eval_default_dino_base.sh $EXPERIMENT_NAME $PRETRAINING_DATA_PATH $KNN_GPU