#-------------------------------------------------------------------------------
# CLASSIFIER TWO SAMPLE TESTS UTILITIES
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains model, data, and helper functions to experiment
#    on the classifier two sample tests.
#-------------------------------------------------------------------------------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import OneClassSVM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd

from constants import *
from shift_dimensionality_reductor import *


#-------------------------------------------------------------------------------
## Models

def lda_binary_classifier(X_train, y_train):
    """
    Build and return the Fisher's linear discriminant analysis classifier.

    :param X_train: the training data (images) - half clean, half shifted.
    :param y_train: the labels (0/ clean or 1/ shifted).
    :param X_test: similar format to X_train, but for evaluation.
    :param y_test: similar format to y_train, but for evaluation.

    :return the scikit lda object.
    """

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    return 

def end_to_end_binary_classifier(dataset, X_train, y_train, X_valid, y_valid, 
    save_path=None):
    """
    End to end neural network-based binary classifier. We used the architecture
    as we defined previously on shift_dimensionality_reductor.py.

    :param dataset: one of Dataset in constants.py.
    :param X_train: the training data (images) - half clean, half shifted.
    :param y_train: the labels (0/ clean, 1/ shifted).
    :param X_valid: validation data of the same format as X_train.
    :param y_valid: validation data of the same format as y_train.
    :param save_path: if specified, will save the trained model on the path.

    :return: trained neural network models.
    """

    orig_dims = X_train.shape[1:] # get the dimension of the image for determining input size

    img_inputs = Input(shape=(orig_dims[0], orig_dims[1], orig_dims[2]))

    # Shared layers (we use similar architecture to the end-to-end)
    x = SharedCNNBlock(dataset)(img_inputs)

    # Output layer
    out = layers.Dense(1, activation="sigmoid")

    model = tf.keras.Model(inputs=img_inputs, outputs=out)

    # Compile and train model
    optimizer = optimizers.Adam(lr=1e-4, amsgrad=True)
    epochs = 200
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    batch_size = 128

    model.compile(loss="binary_crossentropy",
                            metrics=["accuracy"],
                            optimizer=optimizer)

    histories = model.fit(x=X_train, y=y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_valid, y_valid),
                        callbacks=[lr_reducer, early_stopper])

    # Save if specified
    if save_path:
        model.save(save_path)
    
    return histories, model

def one_class_svm(X_train):
    """
    Unsupervised anomaly detection for detecting malicious shifts.

    :param X_train: the training images (half clean, half altered).

    :return: the one class SVM model.
    """

    svm = OneClassSVM()
    svm.fit(X_train)

    return svm

def concept_bottleneck_model():
    # TODO
    pass

def concept_model_extraction():
    # TODO
    pass


#-------------------------------------------------------------------------------
## Dataset construction

def generate_domain_classifier_data(X_train, y_train, X_test, y_test, balanced=True):
    """
    Given two sets of data (x_clean, y_clean) and (x_altered, y_altered), we merge
    them to create a new dataset that will be the input for the domain classifier/
    CS2T.

    :param X_train: the clean (training) data feature
    :param y_train: the clean (training) data label
    :param X_test: the o.o.d/ shifted/ real world (test) data feature
    :param y_test: the o.o.d/ shifted/ real world (test) data label
    :param balanced: if balanced = True, we will downsample the training data so
        that the proportion is the same.

    :return: X_train_new, y_train_new, X_val_new, y_val_new, X_test_new, y_test_new.
    """

    # Balance the training data
    if balanced:
        # Balance the dataset
        if len(X_train) > len(X_test):
            X_train = X_train[:len(X_test)]
            y_train = y_train[:len(y_test)]
        else:
            X_test = X_test[:len(X_train)]
            y_test = y_test[:len(y_train)]

    # Extract half from both sets to be aggregated according to description in the paper
    validation_indices = np.random.choice(X_train.shape[0], 
                    int(np.ceil(X_train.shape[0] * 0.1)), replace=False)
    remaining_indices = [i for i in range(X_train.shape[0]) if i not in validation_indices]
    training_indices = np.random.choice(remaining_indices, 
                int(np.ceil(len(remaining_indices) * 0.5)), replace=False)
    test_indices = [i for i in remaining_indices if i not in training_indices]

    ## Now validation indices will be used to construct the new validation data, training
    # indices will be used to construct the training data, and test indices for test data.

    # Extract datasets
    X_clean_train = X_train[training_indices, :]
    X_clean_test = X_train[test_indices, :]
    X_clean_val = X_train[validation_indices, :]

    ## Altered
    X_altered_train = X_test[training_indices, :]
    X_altered_test = X_test[test_indices, :]
    X_altered_val = X_test[validation_indices, :]

    ## Recombine datasets
    X_train_new = np.append(X_clean_train, X_altered_train, axis=0)
    y_train_new = np.zeros(len(X_train_new))
    y_train_new[len(X_clean_train):] = np.ones(len(X_altered_train))

    X_test_new = np.append(X_clean_test, X_altered_test, axis=0)
    y_test_new = np.zeros(len(X_test_new))
    y_test_new[len(X_clean_test):] = np.ones(len(X_altered_test))

    X_val_new = np.append(X_clean_val, X_altered_val, axis=0)
    y_val_new = np.zeros(len(X_val_new))
    y_val_new[len(X_clean_val):] = np.ones(len(X_altered_val))

    ## Shuffle them
    X_train_new, y_train_new = unison_shuffled_copies(X_train_new, y_train_new)
    X_test_new, y_test_new = unison_shuffled_copies(X_test_new, y_test_new)
    X_val_new, y_val_new = unison_shuffled_copies(X_val_new, y_val_new)

    return X_train_new, y_train_new, X_val_new, y_val_new, x_test_new, y_test_new


#-------------------------------------------------------------------------------
## Helper functions 

def unison_shuffled_copies(a, b):
    """
    Used to shuffle a, b together.

    :param a, b: arrays of same length.

    :return: shuffled a, b
    """

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
