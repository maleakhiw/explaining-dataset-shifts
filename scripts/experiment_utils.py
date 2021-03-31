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
from tqdm import tqdm
import pickle

from shift_applicator import *
from shift_dimensionality_reductor import *
from shift_statistical_test import *
from constants import *

import warnings
warnings.filterwarnings('ignore')


#-------------------------------------------------------------------------------
## Data collection functions

def main_experiment(model, method, X_valid, y_valid, c_valid,
                    X_test, y_test, c_test, shift_type, orig_dims,
                    num_classes, concept_names, concept_values,
                    shift_type_params=None, n_exp=100, n_std=5):
    """
    Calculate the test statistics, p-value, and detection accuracy for a given method
    and shift on all combinations of number of test samples, shift intensities, 
    and proportion of test data that is affected by shift.

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
    :param X_valid, y_valid, c_valid: validation data, which we hypothetically treat as the dataset that we have.
    :param X_test, y_test, c_test: test data, which we hypothetically treat as unseen real-world data, where shift might occur
    :param shift_type: shift type to be applied.
    :param orig_dims: original dimension of the images.
    :param num_classes: number of classes in the original task (y).
    :param concept_names: list of concept names.
    :param concept_values: how many possible concept values for each concept in concept_names (list).
    :param shift_type_params: extra parameters for the shift applicator functions.
    :param n_exp: number of experiments, which we will average on to get shift detection accuracy.
    :param n_exp: number of repeated experiments (e.g., if total_exp = n_std * n_exp). 
        The data for each run is used to compute confidence interval or standard deviation.

    :return: a dictionary containing p-value and detection accuracy for all combination of shift intensities,
        shift proportion, and number of test samples:
        {
            "shift_intensities": {
                "shift_proportion": {
                    "test_samples: : {
                        "test_statistics": [],
                        "p_vals": [],
                        "detection_results": [],
                        "true_detection_results": [], # the true shift detection labels
                        "ppf": [] # inverse cdf for 95% quantile
                    }
                }
            }
        }
    """

    # Possible value of intensities, data proportion affected, test set samples
    shift_intensities = [ShiftIntensity.Small, ShiftIntensity.Medium, ShiftIntensity.Large]
    shift_props = [0.1, 0.5, 1.0] # pre-defined configurations
    test_set_samples = [10, 20, 50, 100, 200, 500, 1000, 10000] # pre-defined configurations

    ## Initialise dictionary used to store result
    dict_result = initialise_result_dictionary(shift_intensities, shift_props, test_set_samples)

    ## Consider all combinations of shift intensities, shift proportion, test samples
    for shift_intensity in tqdm(shift_intensities):
        for shift_prop in shift_props:
            for test_set_sample in test_set_samples:
                # Repeat the experiment n_std and n_exp times for each n_std
                for i in range(n_std):
                    # Create empty list to append results for the given n_std run.
                    dict_result[shift_intensity][shift_prop][test_set_sample]["test_statistics"].append([])
                    dict_result[shift_intensity][shift_prop][test_set_sample]["p_vals"].append([])
                    dict_result[shift_intensity][shift_prop][test_set_sample]["detection_results"].append([])
                    dict_result[shift_intensity][shift_prop][test_set_sample]["true_detection_results"].append([])
                    dict_result[shift_intensity][shift_prop][test_set_sample]["ppf"].append([])

                    # For each experiment run, run n_exp times
                    for j in range(n_exp):
                        # Get test set
                        X_test_subset, y_test_subset, c_test_subset = get_random_data_subset(X_test, y_test, 
                                                                                            c_test, test_set_sample)

                        # Call apply shift method on the test set if do_shift is True,
                        # otherwise, we do not shift. This is used to prevent false
                        # positive. The rate of non-shift examples are determined by rate.
                        # When rate = 1, then all examples are shift examples.
                        shift_rate = 1.0 # change this to determine how many false positive examples to add.
                        do_shift = np.random.uniform(0, 1.0) < shift_rate
                        if do_shift:
                            ## If user wants to apply multiple shift, apply one at a time
                            if isinstance(shift_type, list):
                                X_test_shifted = X_test_subset
                                y_test_shifted = y_test_subset
                                c_test_shifted = c_test_subset
                                for s, p in zip(shift_type, shift_type_params):
                                    X_test_shifted, y_test_shifted, c_test_shifted = apply_shift(X_test_shifted, y_test_shifted, 
                                                                                c_test_shifted, s, p, 
                                                                                shift_intensity, shift_prop)
                            ## Apply only single shift
                            else:
                                X_test_shifted, y_test_shifted, c_test_shifted = apply_shift(X_test_subset, y_test_subset, 
                                                                            c_test_subset, shift_type, shift_type_params, 
                                                                            shift_intensity, shift_prop)
                            
                            # Gold standard label = 1, since we shift the result
                            dict_result[shift_intensity][shift_prop][test_set_sample]["true_detection_results"][i].append(1)
                        # Not do shift as do_shift tell us so (false-positive check)
                        else:
                            X_test_shifted = X_test_subset
                            y_test_shifted = y_test_subset
                            c_test_shifted = c_test_subset

                            # Gold standard label = 0, since we do not shift.
                            dict_result[shift_intensity][shift_prop][test_set_sample]["true_detection_results"][i].append(0)


                        # Perform detection:
                        # 1. Get reduced representation
                        # 2. Perform statistical test
                        test_statistic, p_val, detection_result, ppf = single_experiment(model, method, X_valid, 
                                                                                    X_test_shifted, orig_dims,
                                                                                    num_classes, concept_names, concept_values)

                        # 3. Store result
                        dict_result[shift_intensity][shift_prop][test_set_sample]["test_statistics"][i].append(test_statistic)
                        dict_result[shift_intensity][shift_prop][test_set_sample]["p_vals"][i].append(p_val)
                        dict_result[shift_intensity][shift_prop][test_set_sample]["detection_results"][i].append(detection_result)
                        dict_result[shift_intensity][shift_prop][test_set_sample]["ppf"][i].append(ppf)

    return dict_result

def single_experiment(model, method, X_valid, X_test, 
                orig_dims, num_classes, concept_names, concept_values,
                alpha=0.05):
    """
    This function is called from main_experiment function. Given a single configuration 
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

    :return: (test_statistic, p_val, detection_result, ppf). ppf is the inversed CDF of 95% of the test statistics.
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
        p_val, p_vals, t_vals, ppfs = one_dimensional_test(repr_test, repr_valid, OneDimensionalTest.KS)
        alpha = alpha / repr_valid.shape[1] # Bonferroni correction (divide by number of components)
        if p_val < alpha:
            detection_result = 1 # there is shift
        else:
            detection_result = 0 # no shift found
        
        # Pack result for return
        test_statistic = t_vals
        p_val = p_vals
        ppf = ppfs
        detection_result = detection_result
    
    ## BBSD Argmax
    elif method == DimensionalityReductor.BBSDh:
        repr_valid = np.argmax(model.predict(X_valid), axis=1)
        
        repr_test = np.argmax(model.predict(X_test.reshape(-1, orig_dims[0],
                                                           orig_dims[1],
                                                           orig_dims[2])),
                                                           axis=1)
        
        chi2, p_val, ppf = test_chi2_shift(repr_valid, repr_test, num_classes)

        if p_val < alpha:
            detection_result = 1
        else:
            detection_result = 0
        
        # Pack result for return
        test_statistic = chi2
        p_val = p_val
        detection_result = detection_result
        ppf = ppf
    
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
        ppf_dict = {concept: None for concept in concept_names}

        # Do statistical test for each concept (one dimensional test)
        for concept, repr_valid, repr_test in zip(concept_names, valid_concept_repr, test_concept_repr):
            p_val, p_vals, t_vals, ppfs = one_dimensional_test(repr_valid, repr_test, OneDimensionalTest.KS)
            alpha = alpha / repr_valid.shape[1] # Divided by number of components for Bonferroni correction
            test_statistic[concept] = t_vals
            p_val_dict[concept] = p_vals
            ppf_dict[concept] = ppfs


            if p_val < alpha:
                detection_result[concept] = 1
            else:
                detection_result[concept] = 0
        p_val = p_val_dict
        ppf = ppf_dict
    
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
            test_concept_repr.append(np.argmax(test_preds[i], axis=1))

        # Prepare result
        test_statistic = {concept: None for concept in concept_names}
        p_val_dict = {concept: None for concept in concept_names}
        detection_result = {concept: None for concept in concept_names}
        ppf_dict = {concept: None for concept in concept_names}

        # Do statistical test for each concept (one dimensional test)
        for concept, repr_valid, repr_test, num_concept in zip(concept_names, valid_concept_repr, test_concept_repr, concept_values):
            chi2, p_val, ppf = test_chi2_shift(repr_valid, repr_test, num_concept)
            test_statistic[concept] = chi2
            p_val_dict[concept] = p_val
            ppf_dict[concept] = ppf

            if p_val < alpha:
                detection_result[concept] = 1
            else:
                detection_result[concept] = 0
            
        p_val = p_val_dict
        ppf = ppf_dict
    
    ## PCA
    elif method == DimensionalityReductor.PCA:
        # Valid representation
        X_valid_flatten = X_valid.reshape(X_valid.shape[0], -1)
        repr_valid = model.transform(X_valid_flatten)
        
        # Test representation
        repr_test = model.transform(X_test)
        
        # Do multiple univariate testing
        p_val, p_vals, t_vals, ppfs = one_dimensional_test(repr_test, repr_valid, OneDimensionalTest.KS)
        alpha = alpha / repr_valid.shape[1] # Bonferroni correction (divide by number of components)
        if p_val < alpha:
            detection_result = 1 # there is shift
        else:
            detection_result = 0 # no shift found
        
        # Pack result for return
        test_statistic = t_vals
        p_val = p_vals
        detection_result = detection_result
        ppf = ppfs

    ## SRP
    elif method == DimensionalityReductor.SRP:
        # Valid representation
        X_valid_flatten = X_valid.reshape(X_valid.shape[0], -1)
        repr_valid = model.transform(X_valid_flatten)
        
        # Test representation
        repr_test = model.transform(X_test)
        
        # Do multiple univariate testing
        p_val, p_vals, t_vals, ppfs = one_dimensional_test(repr_test, repr_valid, OneDimensionalTest.KS)
        alpha = alpha / repr_valid.shape[1] # Bonferroni correction (divide by number of components)
        if p_val < alpha:
            detection_result = 1 # there is shift
        else:
            detection_result = 0 # no shift found
        
        # Pack result for return
        test_statistic = t_vals
        p_val = p_vals
        detection_result = detection_result
        ppf = ppfs
    
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
        p_val, p_vals, t_vals, ppfs = one_dimensional_test(repr_test, repr_valid, OneDimensionalTest.KS)
        alpha = alpha / repr_valid.shape[1] # Bonferroni correction (divide by number of components)
        if p_val < alpha:
            detection_result = 1 # there is shift
        else:
            detection_result = 0 # no shift found
        
        # Pack result for return
        test_statistic = t_vals
        p_val = p_vals
        detection_result = detection_result
        ppf = ppfs
    
    return (test_statistic, p_val, detection_result, ppf)

#-------------------------------------------------------------------------------
## Visualisation functions


#-------------------------------------------------------------------------------
## Helper functions

def initialise_result_dictionary(shift_intensities, shift_props, test_set_samples):
    """
    Initialise dictionary used to store result of the experiments.

    :param shift_intensities: all possible shift intensities
    :param shift_props: all possible shift proportions.
    :param test_set_samples: all possible test set samples.

    :return: a dictionary containing p-value and detection accuracy for all combination of shift intensities,
        shift proportion, and number of test samples:
        {
            "shift_intensities": {
                "shift_proportion": {
                    "test_samples: : {
                        "test_statistics": [],
                        "p_vals": [],
                        "detection_results": [],
                        "true_detection_results": [], # the true shift detection labels
                        "ppf": [] # inverse cdf for 95% quantile
                    }
                }
            }
        }
    """

    dict_result = dict()

    ## Generate empty dictionary to store results.
    for shift_intensity in shift_intensities:
        dict_result[shift_intensity] = dict()
        for shift_prop in shift_props:
            dict_result[shift_intensity][shift_prop] = dict()
            for test_set_sample in test_set_samples:
                dict_result[shift_intensity][shift_prop][test_set_sample] = {
                    "test_statistics": [],
                    "p_vals": [],
                    "detection_results": [],
                    "true_detection_results": [],
                    "ppf": []
                }
    
    return dict_result

def get_random_data_subset(X, y, c, test_set_sample):
    """
    Get random (subset) of data of size test_set_sample.

    :param X: the feature/ image.
    :param y: the label.
    :param c: the concept.
    :param test_set_sample: number of sample in the new test set.

    :return subset X, y, c.
    """

    # Random indices
    indices = np.random.choice(X.shape[0], test_set_sample, replace=False)

    # Data subsets
    X_subset = X[indices, :]
    y_subset = y[indices]
    c_subset = c[indices, :]

    return X_subset, y_subset, c_subset

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

def save_result(shift_str, method_str, dict_result, scratch=True):
    """
    Used to save dict result after running the experiment.

    :param shift_str: the shift type name.
    :param method_str: the method name.
    :param dict_result: result to be stored.
    """

    filename = f"{shift_str}_{method_str}.pickle"
    if scratch:
        path = f"/local/scratch/maw219/{filename}"
    else:
        path = f"../../results/dSprites/{filename}"

    with open(path, "wb") as handle:
        pickle.dump(dict_result, handle)
        print("Saving successfully.")

def load_result(shift_str, method_str, scratch=True):
    """
    Used to load pickled experimentation result.

    :param shift_str: indicate the shift type name (string)
    :param method_str: indicate the dimensionality reduction method (string)

    :return: dict_result.
    """

    filename = f"{shift_str}_{method_str}.pickle"
    if scratch:
        path = f"/local/scratch/maw219/{filename}"
    else:
        path = f"../../results/dSprites/{filename}"

    with open(path, "rb") as handle:
        dict_result = pickle.load(handle)
        print("Loading file successfully.")
    
    return dict_result