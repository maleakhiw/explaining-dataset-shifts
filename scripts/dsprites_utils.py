#-------------------------------------------------------------------------------
# DSPRITES UTILITIES
# 
# Author: Maleakhi A. Wijaya
# Description: this file contains functions used to load and preprocess dSprites
#    dataset. The dSprites dataset consists of 2D shapes procedurally generated 
#    from 6 ground truth independent latent factors: 
#        - color (1 possible value)
#        - shape (3 possible values)
#        - scale (6)
#        - rotation (40)
#        - x (32) 
#        - y (32) 
#-------------------------------------------------------------------------------

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from constants import *


#-------------------------------------------------------------------------------
## Load

def load_dsprites(path, dataset_size_used, task, train_size=0.85, class_index=1):
    """
    Load dSprites dataset, split into train, validation, and test sets.

    :param path: the path of the dataset
    :param dataset_size_used: how many instances we will load into RAM
    :param train_size: size of the training set
    :param class_index: 1 for shape

    :return: x_train, x_test, y_train, y_test, c_train, c_test
    """

    # Load dataset
    dataset_zip = np.load(path)

    # Extract relevant datas from the zip file
    imgs = dataset_zip["imgs"] # contains image data (737280 x 64 x 64)
    latents_classes = dataset_zip['latents_classes'] 
        # classification targets (integer index of latents_values)

    # Select data that will be used
    indices_sampled = np.random.randint(0, imgs.shape[0], dataset_size_used)
    X = np.expand_dims(imgs, axis=-1).astype(('float32'))
    
    ## Get the y label, we defined multiple tasks below:
    # TASK 1: predict one of the concept as the end task
    if task == DatasetTask.Task1:
      y = latents_classes[:, class_index][indices_sampled]
    # TASK 2: predict combination of concepts as the end task
    elif task == DatasetTask.Task2:
      y1 = latents_classes[:, class_index[0]][indices_sampled]
      y2 = latents_classes[:, class_index[1]][indices_sampled]
      y = []
      for  i, j in zip(y1, y2):
        y.append(f"{i}_{j}")
      
      # Encode
      le = LabelEncoder()
      y = le.fit_transform(y)
    # TASK 3: predict binary value - we further fine-grained the task by
    # grouping the original labels into two bins.
    else:
      y = latents_classes[:, class_index][indices_sampled]
      n_elements = len(np.unique(y)) // 2 # number of elements in a bin
      bin1_classes = np.random.choice(np.unique(y), n_elements, replace=False)

      new_y = []
      for el in y:
          if el in bin1_classes:
              new_y.append(0)
          else:
              new_y.append(1)
      y = np.array(new_y)
                
    c = latents_classes # concepts
    X = X[indices_sampled]
    c = c[indices_sampled]

    # Split X (image), y (shape for task 1), concepts to train test sets
    x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(X, y, 
        c, train_size=train_size)
    print('Training samples:', x_train.shape[0])
    print('Testing samples:', x_test.shape[0])

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test, c_train, c_test


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
            ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

def visualise_adversarial(original_images, perturb_patterns, adversarial_images, 
                          orig_dims, model, labels_str, n_plots=50):
    """
    Visualise before after adversarial shifts as well as the FGSM pattern.

    :param original_images: list of original images.
    :param perturb_patterns: list of perturb patterns.
    :param adversarial_images: list of adversarial shifted images.
    :param model: the model that we fool.
    :param labels_str: class strings (e.g., ellipse, square, heart for dsprites shapes).
    :param n_plots: number of images to be plotted.
    """

    indices_sampled = np.random.randint(0, original_images.shape[0], n_plots)

    ## Visualise
    for orig_img, perturb_img, adv_img in zip(original_images[indices_sampled], 
                                              perturb_patterns[indices_sampled], 
                                              adversarial_images[indices_sampled]):
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131) # draw original image
        ax2 = fig.add_subplot(132) # draw perturbation pattern
        ax3 = fig.add_subplot(133) # draw adversarial image

        before_predict = model.predict(orig_img.reshape(1, orig_dims[0], orig_dims[1],
                                                                orig_dims[2])).argmax()
        after_predict = model.predict(adv_img.reshape(1, orig_dims[0], orig_dims[1],
                                                                orig_dims[2])).argmax()

        # If channel = 1
        if orig_dims[2] == 1:
            ax1.imshow(orig_img.reshape(orig_dims[0], orig_dims[1]), cmap="gray")
            ax2.imshow(perturb_img.reshape(orig_dims[0], orig_dims[1]), cmap="gray")
            ax3.imshow(adv_img.reshape(orig_dims[0], orig_dims[1]), cmap="gray")
        else:
            ax1.imshow(orig_img.reshape(orig_dims[0], orig_dims[1], orig_dims[2]))
            ax2.imshow(perturb_img.reshape(orig_dims[0], orig_dims[1]))
            ax3.imshow(adv_img.reshape(orig_dims[0], orig_dims[1], orig_dims[2]))

        ax1.set_title(f"Original: {labels_str[before_predict]}", fontsize=14)
        ax2.set_title("FGSM pattern", fontsize=14)
        ax3.set_title(f"Adversarial: {labels_str[after_predict]}", fontsize=14)
        
        # Remove the image axis
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)


#-------------------------------------------------------------------------------
## Helper

def get_latent_sizes():
    """
    Get the size of each concept (possible values of each class).
    """
    
    return np.array([1, 3, 6, 40, 32, 32])


def get_latent_bases():
    """
    Given vector (x, y, z) where each dimension is in base (a, b, c).
    The following function will convert each of (x, y, z) dimensions to decimal.
    """
    
    latent_sizes = get_latent_sizes()
    latent_bases = np.concatenate((latent_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))
    
    return latent_bases


def sample_latent(size=1):
    """
    Used to randomly sample latent of size 'size'. Randomly sample data of size 
    'size'.

    :param size: how many random samples
    
    :return: sample of 'size' latents
    """
    
    latents_sizes = get_latent_sizes()
    samples = np.zeros((size, len(latents_sizes)))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)

    return samples


def latent_to_index(latents):
    """
    Convert from given latent to index position of it in the dataset.

    :param latents: array of latent
    
    :return: list of indices
    """
    
    latents_bases = get_latent_bases()
    return np.dot(latents, latents_bases).astype(int)