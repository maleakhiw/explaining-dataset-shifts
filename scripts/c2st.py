#-------------------------------------------------------------------------------
# CLASSIFIER TWO SAMPLE TESTS
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains functions used for data collection and visualising
#   experimentation results that is related to classifier two sample tests.
# Credit: the most_likely_shifted_samples is adapted from Rabanser's failing-loudly.
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import matplotlib.style as style

from constants import *
from shift_dimensionality_reductor import *
from shift_statistical_test import *

import warnings
warnings.filterwarnings('ignore')
style.use('fivethirtyeight') # Matplotlib style

#-------------------------------------------------------------------------------
## Data collection

#-------------------------------------------------------------------------------
## Visualisation

#-------------------------------------------------------------------------------
## Core functionalities

def most_likely_shifted_samples(c2st_param, model, X_test_flatten, y_test, orig_dims):
    """
    Identify the most likely shifted samples. This is done by inspecting the 
    model prediction confidence when predicting that a given data point comes
    from the target distribution. The more conifdent the prediction, the more
    likely it is shifted.

    :param c2st_param: one of ClassifierTwoSampleTest in constants.py.
    :param model: the trained binary classifier model to make predictions.
    :param X_test_flatten, y_test: image and labels (flattened).
    :param orig_dims: original dimension of the test images (e.g., 64, 64, 3).
    """

    alpha = 0.05

    if c2st_param == ClassifierTwoSampleTest.LDA:
        y_test_pred = model.predict(X_test_flatten)

        y_test_pred_probs = model.predict_proba(X_test_flatten)
        most_conf_test_indices = np.argsort(y_test_pred_probs[:, 1])[::-1]
        most_conf_test_perc = np.sort(y_test_pred_probs[:, 1])[::-1] # score

        # Test whether classification accuracy is statistically significant
        errors = np.count_nonzero(y_test - y_test_pred)
        successes = len(y_test_pred) - errors
        p_val = test_shift_bin(successes, len(y_test_pred), 0.5)
        detection_result = True if p_val < alpha else False # reject null if < 5%

        return most_conf_test_indices, most_conf_test_perc, detection_result, p_val
    
    elif c2st_param == ClassifierTwoSampleTest.FFNN:
        y_test_pred = model.predict(X_test_flatten.reshape(-1, orig_dims[0],
                                                        orig_dims[1], orig_dims[2]))
        
        # Get most anomalous indices sorted in descending order
        most_conf_test_indices = np.argsort(np.squeeze(y_test_pred))[::-1]
        most_conf_test_perc = np.sort(np.squeeze(y_test_pred))[::-1]

        # Test whether classification accuracy is significant
        pred = y_test_pred > 0.5
        errors = np.count_nonzero(y_test - pred)
        successes = len(pred) - errors
        p_val = test_shift_bin(successes, len(pred), 0.5)
        detection_result = True if p_val < alpha else False

        return most_conf_test_indices, most_conf_test_perc, detection_result, p_val
    
    elif c2st_param == ClassifierTwoSampleTest.OCSVM:
        y_test_pred = model.predict(X_test_flatten)
        novelties = X_test_flatten[y_test_pred == -1]

        return novelties, None, -1

    elif c2st_param == ClassifierTwoSampleTest.CBM:
        # If it is sequential or independent, process differently.
        if isinstance(model, ConceptBottleneckModel):
            y_test_pred_probs = model.predict(X_test_flatten.reshape(-1, orig_dims[0],
                                                                    orig_dims[1],
                                                                    orig_dims[2]))
            most_conf_test_indices = np.argsort(y_test_pred_probs[:, 1])[::-1]
            most_conf_test_perc = np.sort(y_test_pred_probs[:, -1])[::-1] # score

            y_test_pred = np.argmax(y_test_pred_probs, axis=1)
            errors = np.count_nonzero(y_test_pred - y_test)
            successes = len(y_test_pred) - errors
            p_val = test_shift_bin(successes, len(y_test_pred), 0.5)
            detection_result = True if p_val < alpha else False
        # For joint, the format is a bit different
        else:
            preds = model.predict(X_test_flatten.reshape(-1, orig_dims[0],
                                                            orig_dims[1], orig_dims[2]))
            y_test_pred = preds[-1]

            # Get most anomalous indices sorted in descending order
            most_conf_test_indices = np.argsort(np.squeeze(y_test_pred))[::-1]
            most_conf_test_perc = np.sort(np.squeeze(y_test_pred))[::-1]

            # Test whether classification accuracy is significant
            pred = y_test_pred > 0.5
            errors = np.count_nonzero(y_test - pred)
            successes = len(pred) - errors
            p_val = test_shift_bin(successes, len(pred), 0.5)
            detection_result = True if p_val < alpha else False

        return most_conf_test_indices, most_conf_test_perc, detection_result, p_val
