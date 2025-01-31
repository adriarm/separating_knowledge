#!/bin/bash
PYTHONPATH=.
EXPERIMENT_NAME=$1
PRETRAINING_DATA_PATH=$2
KNN_GPU=$3
DUMP_FEATURES=True
N_GPUS=8

# Verify number of GPUs matches what was set
VISIBLE_GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

if [ "$VISIBLE_GPU_COUNT" -ne "$N_GPUS" ]; then
    echo "Error: Number of CUDA_VISIBLE_DEVICES ($VISIBLE_GPU_COUNT) does not match N_GPUS ($N_GPUS)."
    echo "Please adjust either CUDA_VISIBLE_DEVICES ($CUDA_VISIBLE_DEVICES) or N_GPUS ($N_GPUS) to ensure they are equal."
    exit 1  # Exit with an error code
fi

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

torchrun --master_port $MASTER_PORT --nproc_per_node=$N_GPUS main_dino.py --arch vit_tiny --use_fp16 True  \
    --data_path $PRETRAINING_DATA_PATH \
    --output_dir encoders/$EXPERIMENT_NAME

mkdir -p encoders/$EXPERIMENT_NAME/checkpoint/knn
if [ ! -f encoders/$EXPERIMENT_NAME/checkpoint/knn/best_acc.txt ]; then
    # only requires 1 GPU:
    echo "Evaluating KNN"
    CUDA_VISIBLE_DEVICES=$KNN_GPU
    EXTRA_ARGS=""
    if [ $DUMP_FEATURES = True ]; then
        echo "Will dump features"
        EXTRA_ARGS="--dump_features encoders/$EXPERIMENT_NAME/checkpoint/knn --dump_nn_images_path encoders/$EXPERIMENT_NAME/checkpoint/knn/nn_neighbors_all"
    fi
    torchrun --master_port $MASTER_PORT --nproc_per_node=1 eval_knn.py --arch vit_tiny --pretrained_weights encoders/$EXPERIMENT_NAME/checkpoint.pth \
        --checkpoint_key teacher --data_path ./data/imagenet --pretraining-epochs 100 \
        --output_dir encoders/$EXPERIMENT_NAME/checkpoint/knn  $EXTRA_ARGS | tee encoders/$EXPERIMENT_NAME/checkpoint/knn/log.txt
else
    echo "KNN evaluation already done"
fi