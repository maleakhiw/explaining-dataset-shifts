#-------------------------------------------------------------------------------
# 3D SHAPES UTILITIES
# 
# Authors: Maleakhi A. Wijaya
# Description: this file contains functions used to load and preprocess the 
#    3dshapes dataset. The 3dshapes dataset is a dataset of 3D shapes procedurally
#    generated from 6 ground truth independent latent factors. These factors are
#    as follows:
#    - floor colour (10 values)
#    - wall colour (10 values)
#    - object colour (10 values)
#    - scale (8 values)
#    - shape (4 values)
#    - orientation (15 values)
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from copy import deepcopy
import pandas as pd

from constants import *
from data_utils import *


#-------------------------------------------------------------------------------
## Load

def load_3dshapes(path, dataset_size_used):
    """
    Load the 3dshapes datasets from the h5 file.

    :param path: the path to the file (including the filename)
    :param dataset_size_used: specify number of instances to be used.

    :return X, c: the images and concepts of size dataset_size_used, dimensions.
    """
    # Load dataset
    dataset = h5py.File(path, 'r')
    # We only consider 100000-300000 due to ram issue.
    # If ram is not an issue (just remove the slicing)
    images = dataset['images'][100000:300000]  # array shape [480000,64,64,3], uint8 in range(256)
    latents_classes = dataset['labels'][100000:300000]  # array shape [480000,6], float64

    # Select data that will be used
    indices_sampled = np.random.randint(0, images.shape[0], dataset_size_used)
    X = images
    c = latents_classes # concepts
    X = X[indices_sampled] / 255. # normalise the image
    c = c[indices_sampled]

    # Label encoder for c
    for i in range(c.shape[1]):
        c[:, i] = LabelEncoder().fit_transform(c[:, i])

    return X, c

def train_test_split_3dshapes(path, dataset_size_used, task, train_size=0.8, class_index=3):
    """
    Split the 3dshapes dataset into training and testing sets.

    :param path: the path to the file (including the filename)
    :param dataset_size_used: specify number of instances to be used.
    :param task: one of DatasetTask in constants.py
    :param train_size: the train data proportion.
    :param class_index: the index of concept that will be used as the end task.
    """

    # Load the data 
    X, c = load_3dshapes(path, dataset_size_used)

    ## Get the y label depending on the task
    # TASK 1: predict one of the concept as the end task.
    if task == DatasetTask.Task1:
        y = c[:, class_index]
    # TASK 2: predict combination of concepts as the end task.
    elif task == DatasetTask.Task2:
        y1 = c[:, class_index[0]]
        y2 = c[:, class_index[1]]

        y = []
        for i, j in zip(y1, y2):
            y.append(f"{i}_{j}")
        
        le = LabelEncoder()
        y = le.fit_transform(y)
    # TASK 3: predict binary value- we further fine-grained the task by grouping
    # the original labels into two bins, chosen randomly.
    else:
        y = c[:, class_index]
        n_elements = len(np.unique(y)) // 2
        bin1_classes = np.random.choice(np.unique(y), n_elements, replace=False)

        new_y = []
        for el in y:
            if el in bin1_classes:
                new_y.append(0)
            else:
                new_y.append(1)
        
        y = np.array(new_y)
    
    # Split image
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, 
        c, train_size=train_size)
    print('Training samples:', X_train.shape[0])
    print('Testing samples:', X_test.shape[0])

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test, c_train, c_test
    

#-------------------------------------------------------------------------------
## Helper

def get_latent_sizes():
    """
    Get the size of each concept (possible value of each class).
    """

    return np.array([10, 10, 10, 8, 4, 15])