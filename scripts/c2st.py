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
from tqdm import tqdm
import matplotlib.style as style
from IPython.display import display, Markdown, Latex

from constants import *
from shift_dimensionality_reductor import *
from shift_statistical_test import *
from experiment_utils import get_random_data_subset
from shift_applicator import *
from c2st_utils import *

import warnings
warnings.filterwarnings('ignore')
style.use('fivethirtyeight') # Matplotlib style

#-------------------------------------------------------------------------------
## Data collection

def domain_classifier_experiment(c2st_param, dataset, X_train, y_train, c_train,
                                X_test, y_test, c_test,
                                shift_type, orig_dims, untrained_cto=None, training_mode=None,
                                n_exp=5, test_sample=10000, shift_type_params=None):
    """
    Calculate the accuracy and confusion matrix for a given method and shift on all
    combinations of shift intensities and proportion of test data affected by shifts.

    :param c2st_param: one of the ClassifierTwoSampleTest parameters in constants.py.
    :param dataset: one of the dataset options in constants.py.
    :param X_train, y_train, c_train: training data (images, labels, concepts). FLATTENED.
    :param X_test, y_test, c_test: testing data (images, labels, concepts), FLATTENED.
    :param shift_type: the shift type, as specified in constants.py.
    :param untrained_cto: untrained concept to output model (by default sklearn's decision
        tree or logistic regression).
    :param training_mode: extra parameter for CBM (which training mode).
    :param n_exp: the number of experiments (the Monte Carlo cross-validation fold).
    :param test_sample: number of samples to be used to fit the classifier.
    :param shift_type_params: extra parameters for shift type.

    :return: a dictionary containing the domain classifier accuracy and confusion matrix.
        The dictionary is of the following form:
    {
        "shift_intensities": {
            "shift_proportion": {
                "accuracy": []
                "confusion_matrix": []
            }
        }
    }
    """

    # Possible value of shift intensities and data proportions affected.
    shift_intensities = [ShiftIntensity.Small, ShiftIntensity.Medium, ShiftIntensity.Large]
    shift_props = [0.1, 0.5, 1.0]

    # Initialise dictionary used to store result
    dict_result = initialise_domain_classifier_dictionary(shift_intensities, shift_props)

    # Train and evaluate model (cross-validation)
    # Store the result in dict_result
    for _ in tqdm(range(n_exp)):
        # No shift case
        X_test_subset, y_test_subset, c_test_subset, indices = get_random_data_subset(X_test, y_test, 
                                                                                            c_test, test_sample)
        
        # Generate synthetic data
        X_train_new, y_train_new, c_train_new, X_val_new, y_val_new, c_val_new, X_test_new, y_test_new, c_test_new = generate_domain_classifier_data(X_train, y_train, c_train, 
                                                                                                                                                    X_test_subset, y_test_subset, c_test_subset)                                                                                                                                            
        # Train the model
        model = build_binary_classifier(dataset, c2st_param, X_train_new, y_train_new,
                                c_train_new, X_val_new, y_val_new, c_val_new, training_mode,
                                untrained_cto, orig_dims)
        # Evaluate the trained model
        acc, cm = evaluate_binary_classifier(c2st_param, model, X_test_new, y_test_new, orig_dims)
        dict_result["no_shift"]["accuracy"].append(acc)
        dict_result["no_shift"]["confusion_matrix"].append(cm)

        for shift_intensity in shift_intensities:
            for shift_prop in shift_props:
                # Select subset of tests randomly (of size as specified in the parameter)
                X_test_subset, y_test_subset, c_test_subset, indices = get_random_data_subset(X_test, y_test, 
                                                                                            c_test, test_sample)
                
                if shift_type == ShiftType.Adversarial:
                    shift_type_params["indices"] = indices # the original indices of X_test wish to be shifted.
                
                X_test_shifted, y_test_shifted, c_test_shifted = apply_shift(X_test_subset, y_test_subset, 
                                                                            c_test_subset, shift_type, shift_type_params, 
                                                                            shift_intensity, shift_prop)
                # Generate synthetic data
                X_train_new, y_train_new, c_train_new, X_val_new, y_val_new, c_val_new, X_test_new, y_test_new, c_test_new = generate_domain_classifier_data(X_train, y_train, c_train, 
                                                                                                                                                            X_test_shifted, y_test_shifted, c_test_shifted)                                                                                                                                            

                # Train the model
                model = build_binary_classifier(dataset, c2st_param, X_train_new, y_train_new,
                                        c_train_new, X_val_new, y_val_new, c_val_new, training_mode,
                                        untrained_cto, orig_dims)

                # Evaluate the trained model
                acc, cm = evaluate_binary_classifier(c2st_param, model, X_test_new, y_test_new, orig_dims)

                # Store results
                dict_result[shift_intensity][shift_prop]["accuracy"].append(acc)
                dict_result[shift_intensity][shift_prop]["confusion_matrix"].append(cm)

    return dict_result

#-------------------------------------------------------------------------------
## Visualisation



#-------------------------------------------------------------------------------
## Add-on functionalities

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


#-------------------------------------------------------------------------------
## Helper functions

def initialise_domain_classifier_dictionary(shift_intensities, shift_props):
    """
    Initialise dictionary for storing domain classifier experiment results.

    :param shift_intensities: all possible shift intensities.
    :param shift_props: all possible shift proportions.

    :return: a dictionary containing accuracy and confusion matrix.
    {
        "shift_intensities": {
            "shift_proportion": {
                "accuracy": []
                "confusion_matrix": []
            }
        }
    }
    """

    dict_result = dict()

    ## Generate empty dictionary to store results
    for shift_intensity in shift_intensities:
        dict_result[shift_intensity] = dict()
        for shift_prop in shift_props:
            dict_result[shift_intensity][shift_prop] = {
                "accuracy": [],
                "confusion_matrix": []
            }
    
    # No shift
    dict_result["no_shift"] = {
        "accuracy": [],
        "confusion_matrix": []
    }
    
    return dict_result

def save_result_dc(shift_str, method_str, dict_result, scratch=True, dataset_fname="dSprites"):
    """
    Used to save dict result after running the experiment.

    :param shift_str: the shift type name.
    :param dict_result: result to be stored.
    """

    filename = f"{shift_str}_dc_{method_str}.pickle"
    if scratch:
        path = f"/local/scratch/maw219/{filename}"
    else:
        path = f"../../results/{dataset_fname}/{filename}"

    with open(path, "wb") as handle:
        pickle.dump(dict_result, handle)
        print("Saving successfully.")

def load_result_dc(shift_str, method_str, scratch=True, dataset_fname="dSprites"):
    """
    Used to load pickled experimentation result.

    :param shift_str: indicate the shift type name (string)

    :return: dict_result.
    """

    filename = f"{shift_str}_dc_{method_str}.pickle"
    if scratch:
        path = f"/local/scratch/maw219/{filename}"
    else:
        path = f"../../results/{dataset_fname}/{filename}"

    with open(path, "rb") as handle:
        dict_result = pickle.load(handle)
        # print("Loading file successfully.")
    
    return dict_result
