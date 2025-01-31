#!/bin/bash

# Define two lists of strings
list1=("imagenet" "places" "shaders_kml_mixup" "shaders_kml" "shaders_mixup" "shaders" "stylegan" "random")
list2=("bloodmnist" "breastmnist" "dermamnist" "octmnist" "organamnist" "organcmnist" "organsmnist" "pathmnist" "pneumoniamnist" "tissuemnist")

# Double for loop to iterate over both lists
for item1 in "${list1[@]}"; do
    for item2 in "${list2[@]}"; do
        echo "$item1 - $item2"
        ./scripts/evals/eval_knn_medicalmnist.sh $item1 $item2
    done
done