#-------------------------------------------------------------------------------
# EXPERIMENT UTILITIES
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains functions used for data collection and visualising
#   experimentation results.
#-------------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


#-------------------------------------------------------------------------------
## Data collection functions

#-------------------------------------------------------------------------------
## Visualisation functions


#-------------------------------------------------------------------------------
## Helper functions

def verify_autoencoder(autoenc, X, n=10):
    """
    Used to visually verify how good the autoencoder can reconstruct the image.

    :param autoenc: autoencoder to be verified.
    :param X: the image data to be reconstructed.
    :param n: the number of image to be displayed.
    """

    plt.figure(figsize=(20, 4))
    decoded_imgs = autoenc(X).numpy()

    for i in range(n):
        # Display original image
        ax = plt.subplot(2, n, i+1)
        plt.imshow(np.squeeze(X[i]))
        plt.title("original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(np.squeeze(decoded_imgs[i]))
        plt.title("reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def unison_shuffled_copies(a, b, c):
    """
    Used to shuffle a, b, c together.

    :param a, b, c: arrays of same length.

    :return: shuffled a, b, c
    """

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]