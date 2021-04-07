#-------------------------------------------------------------------------------
# SMALLNORB UTILITIES
# 
# Authors: Dmitry Kazhdan, Botty Dimanov, Maleakhi A. Wijaya
# Description: this file contains functions used to load and preprocess the 
#    smallNORB dataset. The smallNORB dataset is intended for experiments in 3D
#    object recognition from shape. It contains the following latent factors:
#        - category (5 possible values - original task label)
#        - instance (10)
#        - elevation (9)
#        - azimuth (18)
#        - lighting (6)
#-------------------------------------------------------------------------------

import os
import PIL
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from constants import *


#-------------------------------------------------------------------------------
## Load

def load_smallnorb(files_dir):
    """
    Load smallnorb dataset.

    :param files_dir: path to the actual file.

    :return: X and concepts data.
    """

    filename_template = "smallnorb-{}-{}.mat"
    splits = ["5x46789x9x18x6x2x96x96-training", "5x01235x9x18x6x2x96x96-testing"]
    X_datas, c_datas = [], [] # storage

    for i, split in enumerate(splits):
        # There are three files to be extracted for each splits (see the
        # https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
        data_fname = os.path.join(files_dir, filename_template.format(splits[i], 'dat'))
        cat_fname = os.path.join(files_dir, filename_template.format(splits[i], 'cat'))
        info_fname = os.path.join(files_dir, filename_template.format(splits[i], 'info'))

        X_data = read_binary_matrix(data_fname)
        X_data = resize_images(X_data[:, 0])  # Resize data, and only retain data from 1 camera
        c_cat = read_binary_matrix(cat_fname)
        c_info = read_binary_matrix(info_fname)
        c_info = np.copy(c_info)
        c_info[:, 2:3] = c_info[:, 2:3] / 2  # Set azimuth values to be consecutive digits
        c_data = np.column_stack((c_cat, c_info))

        # Append to the storage
        X_datas.append(X_data)
        c_datas.append(c_data)

    X_data = np.concatenate(X_datas)
    X_data = np.expand_dims(X_data, axis=-1)
    c_data = np.concatenate(c_datas)

    return X_data, c_data

def train_test_split_smallnorb(files_dir, task, train_size=0.75, class_index=0):
    """
    Split the smallnorb dataset into training and testing sets.

    :param files_dir: the path to the actual smallnorb files.
    :param task: one of DatasetTask from constants.py.
    :param train_size: size of the training set
    """

    X_data, c_data = load_smallnorb(files_dir)

    ## Get the y label depending on the tasks:
    # TASK 1: predict one of the concept as the end task
    if task == DatasetTask.Task1:
        y = c_data[:, class_index]
    # TASK 2: predict combination of concepts as the end task
    elif task == DatasetTask.Task2:
        y1 = c_data[:, class_index[0]]
        y2 = c_data[:, class_index[1]]

        y = []
        for i, j in zip(y1, y2):
            y.append(f"{i}_{j}")

        # Encode
        le = LabelEncoder()
        y = le.fit_transform(y)
    # TASK 3: predict binary value - we further fine-grained the task by grouping
    # the original labels into two bins.
    else:
        y = c_data[:, class_index]
        n_elements = len(np.unique(y)) // 2
        bin1_classes = np.random.choice(np.unique(y), n_elements, replace=False)

        new_y = []
        for el in y:
            if el in bin1_classes:
                new_y.append(0)
            else:
                new_y.append(1)
        
        y = np.array(new_y)
    
    c = c_data
    X = X_data
    y = y

    # Split image
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, 
        c, train_size=train_size)
    print('Training samples:', x_train.shape[0])
    print('Testing samples:', x_test.shape[0])

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test, c_train, c_test
    

#-------------------------------------------------------------------------------
## Visualisation

def show_images_grid(imgs_, num_images=25):
    """
    Used to visualise dSprite image in a grid.

    :param imgs_: images to be drawn
    :param num_images: number of images shown in the grid
    """
    
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 2, ncols * 2))
    axes = axes.flatten()
    
    # Draw images on the given grid
    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
        else:
            ax.axis('off')


#-------------------------------------------------------------------------------
## Helper

def get_latent_sizes():
    """
    Get the size of each concept (possible values of each class).
    """

    return np.array([5, 10, 9, 18, 6])

def read_binary_matrix(filename):
    """
    Reads and returns binary formatted matrix stored in filename.
    (Only applicable for the smallnorb data format).

    :param filename: path to the file location.
    """

    with tf.io.gfile.GFile(filename, "rb") as f:
        s = f.read()
        magic = int(np.frombuffer(s, "int32", 1))
        ndim = int(np.frombuffer(s, "int32", 1, 4))
        eff_dim = max(3, ndim)
        raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
        dims = []
        for i in range(0, ndim):
            dims.append(raw_dims[i])

        dtype_map = {
            507333717: "int8",
            507333716: "int32",
            507333713: "float",
            507333715: "double"
        }
        data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data

def resize_images(integer_images):
    """
    Resize the image to 64*64 pixels. Also antialias the image, making it looks
    as original.

    :param integer_images: the images output from read_binary_matrix.
    """
    resized_images = np.zeros((integer_images.shape[0], 64, 64))
    for i in range(integer_images.shape[0]):
        image = PIL.Image.fromarray(integer_images[i, :, :])
        image = image.resize((64, 64), PIL.Image.ANTIALIAS)
        resized_images[i, :, :] = image
    return resized_images / 255.