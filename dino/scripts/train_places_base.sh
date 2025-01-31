#!/bin/bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

eval "$(conda shell.bash hook)"

EXPERIMENT_NAME='places_base'
PRETRAINING_DATA_PATH='../data/places/files/train'
KNN_GPU=6

./scripts/000_train_and_eval_default_dino_base.sh $EXPERIMENT_NAME $PRETRAINING_DATA_PATH $KNN_GPU