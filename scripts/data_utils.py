#-------------------------------------------------------------------------------
# SHARED DATA UTILITIES
# 
# Author: Maleakhi A. Wijaya
# Description: this file contains functions that is shared with the dsprites,
#   smallnorb, and 3dshapes utilities.
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from constants import *


#-------------------------------------------------------------------------------
## Visualisation

def show_images_grid(imgs_, num_images=25):
    """
    Used to visualise images in a grid.

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
            ax.imshow(imgs_[ax_i], cmap='Greys_r', interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
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
            ax2.imshow(perturb_img.reshape(orig_dims[0], orig_dims[1], orig_dims[2]))
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