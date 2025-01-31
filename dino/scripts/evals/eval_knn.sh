
EXPERIMENT_NAME=$1
KNN_GPU=6
DUMP_FEATURES=True

MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

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
    torchrun --master_port $MASTER_PORT --nproc_per_node=1 eval_knn.py --pretrained_weights encoders/$EXPERIMENT_NAME/checkpoint.pth \
        --checkpoint_key teacher --data_path ../data/imagenet \
        --output_dir encoders/$EXPERIMENT_NAME/checkpoint/knn  $EXTRA_ARGS | tee encoders/$EXPERIMENT_NAME/checkpoint/knn/log.txt
else
    echo "KNN evaluation already done"
fi