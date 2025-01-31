# based on:
# https://github.com/Akhilesh64/Image-Classification-using-SIFT/tree/main

#!/usr/bin/env python
# coding: utf-8

##Downloading and unpacking the dataset
#get_ipython().system('wget https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz')
#get_ipython().system('tar -xf mnist_png.tar.gz')

#Importing the required libraries

import os, sys
sys.path.append('..')
sys.path.append('../..')
from my_python_utils.common_utils import *

import cv2
import numpy as np
import time
# import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix




def main(thresh, k, debug=False):
  # list imagenet100
  IMAGENET_PATH='/vision-nfs/torralba/datasets/vision/imagenet100'
  train_image_paths = []
  test_image_paths = []
  train_dirs = listdir(IMAGENET_PATH + '/train', prepend_folder=True)
  train_dirs.sort()
  label_to_class = dict()
  id = 0
  for i, dir in enumerate(tqdm(train_dirs)):
    current_c_files = listdir(dir, prepend_folder=True)
    if debug:
      current_c_files = current_c_files[:10]
    train_image_paths.extend(current_c_files)
    c = dir.split('/')[-1]
    if not c in label_to_class:
      label_to_class[c] = id
      id += 1

  test_dirs = listdir(IMAGENET_PATH + '/val', prepend_folder=True)
  test_dirs.sort()
  for i, dir in enumerate(tqdm(test_dirs)):
    current_c_files = listdir(dir, prepend_folder=True)
    if debug:
      current_c_files = current_c_files[:10]
    test_image_paths.extend(current_c_files)

  train_image_paths.sort()
  test_image_paths.sort()

  t0 = time.time()

  def CalcFeatures(file, th):
    img = cv2.imread(file, 0)
    sift = cv2.xfeatures2d.SIFT_create(th)
    kp, des = sift.detectAndCompute(img, None)
    return des
  
  '''
  All the files appended to the image_path list are passed through the
  CalcFeatures functions which returns the descriptors which are 
  appended to the features list and then stacked vertically in the form
  of a numpy array.
  '''

  parallel = False
  os.makedirs('computed_features_imagenet100', exist_ok=True)

  features_train_path = 'computed_features_imagenet100/features_train_th_{}{}.npy'.format(thresh, '_debug' if debug else '')
  if os.path.exists(features_train_path):
    features_train = load_from_pickle(features_train_path)
  else:
    features_train = process_in_parallel_or_not(lambda x: CalcFeatures(x, thresh), train_image_paths, parallel=parallel)
    dump_to_pickle(features_train_path, features_train)
  # remove Nones
  features_train_paths = [(x, path) for x,path in zip(features_train, train_image_paths) if x is not None]
  features_train = [k[0] for k in features_train_paths]
  train_image_paths = [k[1] for k in features_train_paths]
  all_features_train = np.vstack(features_train)

  features_test_path = 'computed_features/features_test_th_{}{}.npy'.format(thresh, '_debug' if debug else '')
  if os.path.exists(features_test_path):
    features_test = load_from_pickle(features_test_path)
  else:
    features_test = process_in_parallel_or_not(lambda x: CalcFeatures(x, thresh), test_image_paths, parallel=parallel)
    dump_to_pickle(features_test_path, features_test)
  
  '''
  K-Means clustering is then performed on the feature array obtained 
  from the previous step. The centres obtained after clustering are 
  further used for bagging of features.
  '''

  k = 150
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
  flags = cv2.KMEANS_RANDOM_CENTERS
  compactness, labels, centres = cv2.kmeans(all_features_train, k, None, criteria, 10, flags)

  '''
  The bag_of_features function assigns the features which are similar
  to a specific cluster centre thus forming a Bag of Words approach.  
  '''

  def bag_of_features(features, centres, k = 500):
      vec = np.zeros((1, k))
      for i in range(features.shape[0]):
          feat = features[i]
          diff = np.tile(feat, (k, 1)) - centres
          dist = pow(((pow(diff, 2)).sum(axis = 1)), 0.5)
          idx_dist = dist.argsort()
          idx = idx_dist[0]
          vec[0][idx] += 1
      return vec

  labels_train = []
  vec_train = []
  for file_name, img_des in tqdm(list(zip(train_image_paths, features_train))):
    if img_des is not None:
      img_vec = bag_of_features(img_des, centres, k)
      vec_train.append(img_vec)
      labels_train.append(label_to_class[file_name.split('/')[-2]])
  vec_train = np.vstack(vec_train)

  labels_test = []
  vec_test = []
  for file_name, img_des in tqdm(list(zip(test_image_paths, features_test))):
    if img_des is not None:
      img_vec = bag_of_features(img_des, centres, k)
      vec_test.append(img_vec)
      labels_test.append(label_to_class[file_name.split('/')[-2]])
  vec_test = np.vstack(vec_test)

  '''
  Splitting the data formed into test and split data and training the 
  SVM Classifier.
  '''

  # X_train, X_test, y_train, y_test = train_test_split(vec, labels, test_size=0.2)
  X_train, X_test, y_train, y_test = vec_train, vec_test, labels_train, labels_test
  clf = SVC()
  clf.fit(X_train, y_train)
  preds = clf.predict(X_test)
  acc = accuracy_score(y_test, preds)
  conf_mat = confusion_matrix(y_test, preds)

  t1 = time.time()
  
  return acc*100, conf_mat, (t1-t0)

if __name__ == '__main__':
  debug = False
  for th in range(5,100,5):
    for video_reader in [150, 256, 512, 1024, 2048, 4096]:
      acc_file = 'resultsimagenet_100/acc_th_{}_k_{}{}.txt'.format(th, video_reader, '_debug' if debug else '')
      os.makedirs('results', exist_ok=True)
      if os.path.exists(acc_file):
        print("Already computed for th={}, k={}, with accuracy:".format(th, video_reader))
        print(read_text_file(acc_file))
        continue
      print('\nCalculating for a threshold of {}, k={}'.format(th, video_reader))
      data = main(th, video_reader, debug=debug)
      accuracy = data[0]
      conf_mat = data[1]
      timer = data[2]
      print('\nAccuracy = {} %\nTime taken = {} sec'.format(accuracy, timer))
      write_text_file_lines([float2str(accuracy, 2)], acc_file)

