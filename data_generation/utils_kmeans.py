import time

import numpy as np
from matplotlib import pyplot as plt

from pykeops.numpy import LazyTensor
import pykeops.config

dtype = "float32"  # May be 'float32' or 'float64'

def KMeans(x, rnggen, K=10, Niter=10, verbose=False):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    # c = np.copy(x[:K, :])  # Simplistic random initialization
    # idx = rnggen.choice(N, K)
    # c = np.copy(x[idx, :])  # Simplistic random initialization
    # c += 0.1 * rnggen.standard_normal(c.shape)
    c = np.copy(np.unique(x, axis=0)[:K])
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):
        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(
            -1
        )  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(axis=1).astype(int).reshape(N)  # Points -> Nearest cluster

        Ncl = np.bincount(cl).astype(dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with np.bincount:
            c[:, d] = np.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print(
            "K-means example with {:,} points in dimension {:,}, K = {:,}:".format(
                N, D, K
            )
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

def SoftKMeans(x, rnggen, K=10, Niter=10, verbose=False):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    # c = np.copy(x[:K, :])  # Simplistic random initialization
    # idx = rnggen.choice(N, K)
    # c = np.copy(x[idx, :])  # Simplistic random initialization
    # c += 0.1 * rnggen.standard_normal(c.shape)
    c = np.copy(np.unique(x, axis=0)[:K])
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):
        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(
            -1
        )  # (Npoints, Nclusters) symbolic matrix of squared distances
        # cl = D_ij.argmin(axis=1).astype(int).reshape(N)  # Points -> Nearest cluster
        K_ij = (-D_ij).exp()
        cl = K_ij / K_ij.sum(axis=0)

        # Ncl = np.bincount(cl).astype(dtype)  # Class weights
        # for d in range(D):  # Compute the cluster centroids with np.bincount:
        #     c[:, d] = np.bincount(cl, weights=x[:, d]) / Ncl
        c = (x_i * K_ij).sum(axis=0) / K_ij.sum(axis=0)
        # print('c calculated', c.shape)
        # assert False

    end = time.time()

    if verbose:
        print(
            "K-means example with {:,} points in dimension {:,}, K = {:,}:".format(
                N, D, K
            )
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

def Fast2Means(x, rnggen, Niter=10, verbose=False):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    # start = time.time()
    # c = np.copy(x[:K, :])  # Simplistic random initialization
    # idx = rnggen.choice(N, K)
    # c = np.copy(x[idx, :])  # Simplistic random initialization
    # c += 0.1 * rnggen.standard_normal(c.shape)
    u = np.unique(x, axis=0)
    c = np.array([u[0], u[-1]])
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):
        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(
            -1
        )  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(axis=1).astype(int).reshape(N)  # Points -> Nearest cluster

        Ncl = np.bincount(cl).astype(dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with np.bincount:
            c[:, d] = np.bincount(cl, weights=x[:, d]) / Ncl

    return cl, c
    