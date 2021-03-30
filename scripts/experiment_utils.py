#-------------------------------------------------------------------------------
# EXPERIMENT UTILITIES
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains functions used for data collection and visualising
#   experimentation results.
#-------------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


#-------------------------------------------------------------------------------
## Helper functions

def unison_shuffled_copies(a, b, c):
    """
    Used to shuffle a, b, c together.

    :param a, b, c: arrays of same length.

    :return: shuffled a, b, c
    """

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]