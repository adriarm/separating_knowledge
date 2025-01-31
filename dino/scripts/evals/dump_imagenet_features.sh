
EXPERIMENT_NAME=$1
ARCH=vit_small
KNN_GPU=5

MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))


export CUDA_VISIBLE_DEVICES=$KNN_GPU
mkdir -p encoders/$EXPERIMENT_NAME/checkpoint/imagenet_dumps
torchrun --master_port $MASTER_PORT --nproc_per_node=1 dump_imagenet_features.py --pretrained_weights encoders/$EXPERIMENT_NAME/checkpoint.pth --arch $ARCH \
    --checkpoint_key teacher | tee -a encoders/$EXPERIMENT_NAME/checkpoint/imagenet_dumps/log.txt