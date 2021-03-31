#-------------------------------------------------------------------------------
# EXPERIMENT UTILITIES
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains functions used for data collection and visualising
#   experimentation results.
#-------------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from shift_applicator import *
from shift_dimensionality_reductor import *
from shift_statistical_test import *
from constants import *

import warnings
warnings.filterwarnings('ignore')


#-------------------------------------------------------------------------------
## Data collection functions

def single_experiment(model, method, X_valid, X_test, 
                orig_dims, num_classes, concept_names, concept_values,
                alpha=0.05):
    """
    This function is called from main_pipeline function. Given a single configuration 
    (e.g., dimensionality reduction method, validation, and test data, 
    this function calculates p-value, test statistics, and determine if shift exists. 
    
    :param model: the dimensionality reduction that has been fitted with source data.
    :param method: the shift detection method which can be one of those defined
        in DimensionalityReductor in constants.py.
        - 'BBSDs': softmax label classifier (BBSD)
        - 'BBSDh': argmax/ hard prediction label classifier 
        - 'CBSDs': softmax on the concept layer 
        - 'CBSDh': argmax/ hard prediction label classifier
        - 'PCA': used to reduce the dimension of X_valid and X_test
        - 'SRP': used to reduce the dimension of X_valid and X_test
        - 'UAE': Autoencoder based method to reduce the dimension of X_valid and X_test
        - 'TAE': Autoencoder based method to reduce the dimension of X_valid and X_test
    :param X_valid: validation data, which we hypothetically treat as the dataset that we have.
    :param X_test: test data, which we hypothetically treat as unseen real-world data, where shift might occur
    :param orig_dims: original dimension of the images.
    :param num_classes: number of classes in the original task (y).
    :param concept_names: list of concept names.
    :param concept_values: how many possible concept values for each concept in concept_names (list).
    :param alpha: significance test value.

    :return: (test_statistic, p_val, detection_result)
    """

    ## BBSD Softmax
    if method == DimensionalityReductor.BBSDs:
        # Valid representation
        repr_valid = model.predict(X_valid)
        
        # Test representation
        # Note: need to reshape test first as it is flatten previously
        repr_test = model.predict(X_test.reshape(-1, orig_dims[0], 
                                                 orig_dims[1],
                                                 orig_dims[2]))
        
        # Do multiple univariate testing
        p_val, p_vals, t_vals = one_dimensional_test(repr_test, repr_valid)
        alpha = alpha / repr_valid.shape[1] # Bonferroni correction (divide by number of components)
        if p_val < alpha:
            detection_result = 1 # there is shift
        else:
            detection_result = 0 # no shift found
        
        # Pack result for return
        test_statistic = t_vals
        p_val = p_vals
        detection_result = detection_result
    
    ## BBSD Argmax
    elif method == DimensionalityReductor.BBSDh:
        repr_valid = np.argmax(model.predict(X_valid), axis=1)
        
        repr_test = np.argmax(model.predict(X_test.reshape(-1, orig_dims[0],
                                                           orig_dims[1],
                                                           orig_dims[2])),
                                                           axis=1)
        
        chi2, p_val = test_chi2_shift(repr_valid, repr_test, num_classes)

        if p_val < alpha:
            detection_result = 1
        else:
            detection_result = 0
        
        # Pack result for return
        test_statistic = chi2
        p_val = p_val
        detection_result = detection_result
    
    ## CBSD softmax
    elif method == DimensionalityReductor.CBSDs:
        # Valid representation
        preds = model.predict(X_valid)

        # Get valid concept representations
        valid_concept_repr = []
        for i in range(len(concept_names)):
            valid_concept_repr.append(preds[i])
        
        # Test representation
        test_preds = model.predict(X_test.reshape(-1, orig_dims[0],
                                                  orig_dims[1],
                                                  orig_dims[2]))

        # Get test concept representations
        test_concept_repr = []
        for i in range(len(concept_names)):
            test_concept_repr.append(test_preds[i])
        
        # Prepare result
        test_statistic = {concept: None for concept in concept_names}
        p_val_dict = {concept: None for concept in concept_names}
        detection_result = {concept: None for concept in concept_names}

        # Do statistical test for each concept (one dimensional test)
        for concept, repr_valid, repr_test in zip(concept_names, valid_concept_repr, test_concept_repr):
            p_val, p_vals, t_vals = one_dimensional_test(repr_valid, repr_test)
            alpha = alpha / repr_valid.shape[1] # Divided by number of components for Bonferroni correction
            test_statistic[concept] = t_vals
            p_val_dict[concept] = p_vals

            if p_val < alpha:
                detection_result[concept] = 1
            else:
                detection_result[concept] = 0
        p_val = p_val_dict
    
    ## CBSD argmax
    elif method == DimensionalityReductor.CBSDh:
        # Valid representation
        preds = model.predict(X_valid)
        valid_concept_repr = []
        for i in range(len(concept_names)):
            valid_concept_repr.append(np.argmax(preds[i], axis=1))
        
        # Test representation
        test_preds = model.predict(X_test.reshape(-1, orig_dims[0],
                                                  orig_dims[1],
                                                  orig_dims[2]))
        test_concept_repr = []
        for i in range(len(concept_names)):
            test_concept_repr.append(np.argmax(preds[i], axis=1))

        # Prepare result
        test_statistic = {concept: None for concept in concept_names}
        p_val_dict = {concept: None for concept in concept_names}
        detection_result = {concept: None for concept in concept_names}

        # Do statistical test for each concept (one dimensional test)
        for concept, repr_valid, repr_test, num_concept in zip(concept_names, valid_concept_repr, test_concept_repr, concept_values):
            chi2, p_val = test_chi2_shift(repr_valid, repr_test, num_concept)
            test_statistic[concept] = chi2
            p_val_dict[concept] = p_val

            if p_val < alpha:
                detection_result[concept] = 1
            else:
                detection_result[concept] = 0
            
        p_val = p_val_dict
    
    ## PCA
    elif method == DimensionalityReductor.PCA:
        # Valid representation
        X_valid_flatten = X_valid.reshape(X_valid.shape[0], -1)
        repr_valid = model.transform(X_valid_flatten)
        
        # Test representation
        repr_test = model.transform(X_test)
        
        # Do multiple univariate testing
        p_val, p_vals, t_vals = one_dimensional_test(repr_test, repr_valid)
        alpha = alpha / repr_valid.shape[1] # Bonferroni correction (divide by number of components)
        if p_val < alpha:
            detection_result = 1 # there is shift
        else:
            detection_result = 0 # no shift found
        
        # Pack result for return
        test_statistic = t_vals
        p_val = p_vals
        detection_result = detection_result

    ## SRP
    elif method == DimensionalityReductor.SRP:
        # Valid representation
        X_valid_flatten = X_valid.reshape(X_valid.shape[0], -1)
        repr_valid = model.transform(X_valid_flatten)
        
        # Test representation
        repr_test = model.transform(X_test)
        
        # Do multiple univariate testing
        p_val, p_vals, t_vals = one_dimensional_test(repr_test, repr_valid)
        alpha = alpha / repr_valid.shape[1] # Bonferroni correction (divide by number of components)
        if p_val < alpha:
            detection_result = 1 # there is shift
        else:
            detection_result = 0 # no shift found
        
        # Pack result for return
        test_statistic = t_vals
        p_val = p_vals
        detection_result = detection_result
    
    ## UAE and TAE
    elif method in {DimensionalityReductor.UAE, DimensionalityReductor.TAE}:
        # Valid representation
        repr_valid = model(X_valid)
        
        # Test representation
        repr_test = model(X_test.reshape(-1, orig_dims[0],
                                            orig_dims[1],
                                            orig_dims[2]))
        
        # Flatten autoencoder result
        repr_valid = repr_valid.numpy().reshape(repr_valid.shape[0], -1)
        repr_test = repr_test.numpy().reshape(repr_test.shape[0], -1)
        
        # Do multiple univariate testing
        p_val, p_vals, t_vals = one_dimensional_test(repr_test, repr_valid)
        alpha = alpha / repr_valid.shape[1] # Bonferroni correction (divide by number of components)
        if p_val < alpha:
            detection_result = 1 # there is shift
        else:
            detection_result = 0 # no shift found
        
        # Pack result for return
        test_statistic = t_vals
        p_val = p_vals
        detection_result = detection_result
    
    return (test_statistic, p_val, detection_result)

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
        plt.imshow(np.squeeze(X[i]), cmap="gray")
        plt.title("original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(np.squeeze(decoded_imgs[i]), cmap="gray")
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