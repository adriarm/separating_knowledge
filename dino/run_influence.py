import os
import sys
import argparse

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

import ipdb

print(torch.__version__)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

def compute_influence_distribution(args):
    # Dataset and transform
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder('/vision-nfs/torralba/datasets/vision/imagenet/train', transform=transform)
    test_dataset = datasets.ImageFolder('/vision-nfs/torralba/datasets/vision/imagenet/val', transform=transform)

    # Path
    path = args.path
    data_path = '/data/vision/torralba/datasets/imagenet_pytorch_new'

    # Features
    train_features = torch.load(os.path.join(path, "trainfeat.pth"))
    test_features = torch.load(os.path.join(path, "testfeat.pth"))
    train_labels = torch.load(os.path.join(path, "trainlabels.pth"))
    test_labels = torch.load(os.path.join(path, "testlabels.pth"))

    dataset = ReturnIndexDataset(os.path.join(data_path, 'train'), transform=None)
    train_filenames = [k[0] for k in dataset.samples]
    labels_to_test = [k[1] for k in dataset.samples]
    labels_from_features = train_labels
    assert (np.array(labels_to_test) == np.array(labels_from_features.cpu())).all(), "Test labels and test labels from dataset are not the same, the dataset is processed shuffled."

    dataset = ReturnIndexDataset(os.path.join(data_path, 'val'), transform=None)
    test_filenames = [k[0] for k in dataset.samples]
    labels_to_test = [k[1] for k in dataset.samples]
    labels_from_features = test_labels
    assert (np.array(labels_to_test) == np.array(labels_from_features.cpu())).all(), "Test labels and test labels from dataset are not the same, the dataset is processed shuffled."

    # Iterate over test examples
    num_test_images = test_labels.shape[0]
    retrieval_one_hot = torch.zeros(args.k+1, args.num_classes).to(train_features.device)
    batch_size = 1
    train_features = train_features.t()

    influent_examples_good = []
    influent_examples_bad = []
    top1, total = 0.0, 0
    for idx in tqdm(range(num_test_images)):
        features = test_features[idx:idx+1]
        targets = test_labels[idx:idx+1]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(args.k+1, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)
        #print('candidates', candidates.shape)

        retrieval_one_hot.resize_(batch_size * args.k+1, args.num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(args.T).exp_()

        # Probs with k and k+1 neighbours
        probs_k = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, args.num_classes)[:, :-1, :], # use only args.k neighbours for probs
                distances_transform.view(batch_size, -1, 1)[:, :-1, :], 
            ),
            1,
        )
        probs_kplus1 = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, args.num_classes), # use only args.k neighbours for probs
                distances_transform.view(batch_size, -1, 1), 
            ),
            1,
        )
        #print('probs_k', probs_k.shape)
        #print('probs_kplus1', probs_kplus1.shape)

        # Obtain threshold
        probs_k, predictions_k = probs_k.sort(1, True)
        pred = predictions_k[0, 0]

        probs_kplus1[:, pred] = -torch.inf
        probs_kplus1, predictions_kplus1 = probs_kplus1.sort(1, True)
        second_best_pred = predictions_kplus1[0, 0]
        
        thresh = probs_k[0, 0] - probs_kplus1[0, 0]
        if candidates[0, -1] == pred:
            thresh += distances_transform[0, args.k]

        influent_examples_idx_mask = np.logical_and(distances_transform[:, :args.k] > thresh, retrieved_neighbors[:, :args.k] == pred)
        influent_examples_idx = indices[:, :-1][influent_examples_idx_mask]
        influent_examples_idx = list(influent_examples_idx)
        influent_examples_idx = [(idx, i) for i in influent_examples_idx]
        if pred == targets[0]:
            influent_examples_good.extend(influent_examples_idx)
        if pred != targets[0] and second_best_pred == targets[0]:
            influent_examples_bad.extend(influent_examples_idx)

        correct = predictions_k.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)
        # ipdb.set_trace()
    # ipdb.set_trace()

    # influence_counts_good = 100 * (torch.unique(torch.cat([torch.LongTensor(influent_examples_good), torch.arange(train_labels.shape[0])]), return_counts=True)[1] - 1) / total
    # influence_counts_bad = 100 * (torch.unique(torch.cat([torch.LongTensor(influent_examples_bad), torch.arange(train_labels.shape[0])]), return_counts=True)[1] - 1) / total
    influence_counts_good = torch.LongTensor(influent_examples_good)
    influence_counts_bad = torch.LongTensor(influent_examples_bad)
    top1 = top1 * 100.0 / total

    return top1, influence_counts_good, influence_counts_bad


from torchvision import datasets
class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

# KNN classifier
import random

@torch.no_grad()
def knn_classifier(train_features, train_labels, train_filenames, test_features, test_labels, test_filenames, k, T, num_classes=1000, dump_images_path=None, n_to_dump=500):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = max(num_test_images // num_chunks, 1)
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    if not dump_images_path is None:
        indices_to_plot = list(range(len(test_labels)))
        # to always plot the same accross experiments for easy comparison
        random.seed(1338)
        random.shuffle(indices_to_plot)
        indices_to_plot = set(indices_to_plot[:n_to_dump])

    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )


        _, predictions = probs.sort(1, True)
        # for plot_idx in range(idx, min(imgs_per_chunk + idx, num_test_images)):
            # if dump_images_path is not None and plot_idx in indices_to_plot:
            #     test_image_filename = test_filenames[plot_idx]
            #     output_image_path = os.path.join(dump_images_path, test_image_filename.split('/')[-1])
            #     if not os.path.exists(output_image_path) and k >= 5:
            #         os.makedirs(dump_images_path, exist_ok=True)
            #         #test_image = utils.cv2_imread(test_image_filename)
            #         neighbor_files = [train_filenames[indices[plot_idx - idx, i]] for i in range(5)]
            #         #neighbor_images = [utils.cv2_imread(f) for f in neighbor_files]
            #         #output_image = utils.tile_images([test_image] + neighbor_images, (6,1))
            #         #utils.cv2_imwrite(output_image, output_image_path)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Influence function computation')
    parser.add_argument('--path', help='Path to save logs and checkpoints')
    parser.add_argument('--k', default=20, type=int, help='Num neighbours')
    parser.add_argument('--T', default=0.07, type=float, help='Number of data loading workers per GPU.')
    parser.add_argument('--num_classes', default=1000, type=int, help='Number of data loading workers per GPU.')
    args = parser.parse_args()

    top1, influence_counts_good, influence_counts_bad = compute_influence_distribution(args)

    os.makedirs(os.path.join(args.path, 'run_influence'), exist_ok=True)
    # with open(os.path.join(args.path, 'run_influence', 'top1.txt'), 'w') as f:
    #     f.write(f"{top1:.2f}")
    torch.save(influence_counts_good, os.path.join(args.path, 'run_influence', 'influence_raw_good.pth'))
    torch.save(influence_counts_bad, os.path.join(args.path, 'run_influence', 'influence_raw_bad.pth')) 