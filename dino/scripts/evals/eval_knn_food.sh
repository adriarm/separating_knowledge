EXPERIMENT_NAME=$1
ARCH=vit_small
KNN_GPU=6
DUMP_FEATURES=True

MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

mkdir -p encoders/$EXPERIMENT_NAME/checkpoint/knn_food101
if [ ! -f encoders/$EXPERIMENT_NAME/checkpoint/knn_food101/best_acc.txt ]; then
    # only requires 1 GPU:
    echo "Evaluating KNN"
    CUDA_VISIBLE_DEVICES=$KNN_GPU
    EXTRA_ARGS=""
    if [ $DUMP_FEATURES = True ]; then
        echo "Will dump features"
        EXTRA_ARGS="--dump_features encoders/$EXPERIMENT_NAME/checkpoint/knn_food101 --dump_nn_images_path encoders/$EXPERIMENT_NAME/checkpoint/knn_food101/nn_neighbors_all"
    fi
    torchrun --master_port $MASTER_PORT --nproc_per_node=1 eval_knn.py --pretrained_weights encoders/$EXPERIMENT_NAME/checkpoint.pth --arch $ARCH \
        --checkpoint_key teacher --data_path ../data/food-101/food-101 \
        --output_dir encoders/$EXPERIMENT_NAME/checkpoint/knn_food101  $EXTRA_ARGS | tee encoders/$EXPERIMENT_NAME/checkpoint/knn_food101/log.txt
else
    echo "KNN evaluation already done"
fi