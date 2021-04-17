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

def summary_tables_dc(list_dict_result, list_labels):
    """
    Generate summary table displaying the accuracy for determining whether data
    instances come from source or target distributions. We marginalise based on
    shift intensities.

    :param list_dict_result: list of dictionary results from experimentation (see
        domain_classifier_experiment for the dictionary structure).
    :param list_labels: list method names (to be used for printing purposes).
    """

    shift_intensities = [ShiftIntensity.Small, ShiftIntensity.Medium, ShiftIntensity.Large]
    shift_props = [0.1, 0.5, 1.0]

    # Iterate over all methods and print tables
    for dict_result, label in zip(list_dict_result, list_labels):
        display(Markdown(f"## Method: {label}"))

        # print(f"No-shift accuracy: {round(np.mean(dict_result['no_shift']['accuracy']), 2)}")

        # [[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]
        accuracy = []

        for shift_intensity in shift_intensities:
            temp = []
            for shift_prop in shift_props:
                acc = np.mean(dict_result[shift_intensity][shift_prop]["accuracy"])
                temp.append(round(acc, 2))
            accuracy.append(temp)
        
        # Display table
        accuracy_df = pd.DataFrame(accuracy)
        accuracy_df.index = ["Small", "Medium", "Large"]
        accuracy_df.columns = ["10%", "50%", "100%"]
        display(accuracy_df)
        print()

def barplot_accuracy_domain_classifier(list_dict_result, list_labels):
    """
    Create a bar plot depicting the accuracy of various methods for 
    each shift type and intensity. (y-axis=accuracy, x-axis=intensity), each method = bars
    with different colour. title = shift proportion (3 axes: 10%, 50%, and 100%).

    :param list_dict_result: list of dictionary results from experimentation (see
        domain_classifier_experiment for the dictionary structure).
    :param list_labels: list method names (to be used as legend).
    """

    ## The following are the data type that is accepted by the plotting function.
    # Dataframe
    # acc | method (hue) | intensity | proportion
    # Dict
    # {
        # "acc": [],
        # "method": [],
        # "intensity": [],
        # "proportion": []
    # }

    dict_plot = {
        "accuracy": [],
        "method": [],
        "intensity": [],
        "proportion": []
    }

    shift_intensities = [ShiftIntensity.Small, ShiftIntensity.Medium, ShiftIntensity.Large]
    shift_intensities_str = ["small", "medium", "large"]
    shift_props = [0.1, 0.5, 1.0]

    # Iterate over all methods and print tables
    for dict_result, label in zip(list_dict_result, list_labels):
        for shift_intensity_str, shift_intensity in zip(shift_intensities_str, shift_intensities):
            for shift_prop in shift_props:
                for i in range(len(dict_result[shift_intensity][shift_prop]["accuracy"])):
                    acc = dict_result[shift_intensity][shift_prop]["accuracy"][i]
                    dict_plot["accuracy"].append(acc)
                    dict_plot["method"].append(label)
                    dict_plot["intensity"].append(shift_intensity_str)
                    dict_plot["proportion"].append(shift_prop)

    df_dict_plot = pd.DataFrame(dict_plot)
    df_dict_plot["proportion"] = df_dict_plot["proportion"].map({0.1: "10%", 0.5: "50%", 1.0: "100%"})

    # Draw the bar plot
    g = sns.catplot(x="intensity", y="accuracy", hue="method", col="proportion",
        data=df_dict_plot, kind="bar", height=4, aspect=.8, ci=95, errwidth=2)
    g.set(ylim=(0, 1.0))
    
    (g.set_axis_labels("Intensity", "Accuracy")
      .set_xticklabels(["Small", "Medium", "Large"])
      .set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0]))

    g.fig.set_facecolor("white")
    for ax in g.axes[0]:
        ax.set_facecolor("white")
        ax.tick_params(axis="both", which="major", labelsize=12)
        
        # Despine
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.title.set_fontsize(16)
    
    # Edit legend
    g._legend.set_title("Methods")
    new_labels = ["LDA", "FFNN", "CBM (Independent)", "CBM (Sequential)", "CBM (Joint)"]
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

    plt.show()


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
