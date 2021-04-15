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
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from constants import *
from shift_dimensionality_reductor import *


#-------------------------------------------------------------------------------
## Main domain classifier

def build_binary_classifier(dataset, c2st_param, X_train, y_train, c_train, X_valid, 
                        y_valid, c_valid, training_mode, untrained_cto, orig_dims):
    """
    Given the c2st_param (specification of the domain classifier to be used),
    this function will build and train the appropriate models.

    :param dataset: one of Dataset in constants.py.
    :param c2st_param: one of ClassifierTwoSampleTest.
    :param X_train, y_train, c_train: the training data (flattened).
    :param X_valid, y_valid, c_valid: the validation data (flattened).
    :param training_mode: one of ConceptBottleneckTraining.
    :param untrained_cto: untrained concept-to-output model - used for CBM.
    :param orig_dims: original dimension of the image (e.g., 64 x 64 x 3).

    :return: the trained binary classifier model.
    """

    if c2st_param == ClassifierTwoSampleTest.LDA:
        return lda_binary_classifier(X_train, y_train)
    
    elif c2st_param == ClassifierTwoSampleTest.FFNN:
        X_train = X_train.reshape(-1, orig_dims[0], orig_dims[1], orig_dims[2])
        X_valid = X_valid.reshape(-1, orig_dims[0], orig_dims[1], orig_dims[2])

        _, model = end_to_end_binary_classifier(dataset, X_train, y_train, X_valid, y_valid)
        return model
    
    elif c2st_param == ClassifierTwoSampleTest.CBM:
        X_train = X_train.reshape(-1, orig_dims[0], orig_dims[1], orig_dims[2])
        X_valid = X_valid.reshape(-1, orig_dims[0], orig_dims[1], orig_dims[2])

        model = cbm_binary_classifier(dataset, training_mode, X_train, c_train, y_train, 
            X_valid, c_valid, y_valid, untrained_cto, path=None)
        return model

def evaluate_binary_classifier(c2st_param, model, X_test, y_test, orig_dims):
    """
    Given a trained model, this function evaluate its performance.

    :param c2st_param: one of ClassifierTwoSampleTest from constants.py.
    :param model: trained model.
    :param X_test, y_test: flattened test data.

    :return: accuracy score and confusion matrix
    """

    if c2st_param == ClassifierTwoSampleTest.LDA:
        y_test_pred = model.predict(X_test)
        acc = accuracy_score(y_test_pred, y_test)
        cm = confusion_matrix(y_test_pred, y_test)
    
    elif c2st_param == ClassifierTwoSampleTest.FFNN:
        y_test_pred = model.predict(X_test.reshape(-1, orig_dims[0],
                                            orig_dims[1], orig_dims[2]))
        pred = y_test_pred > 0.5
        acc = accuracy_score(pred, y_test)
        cm = confusion_matrix(pred, y_test)

    elif c2st_param == ClassifierTwoSampleTest.CBM:
        # If it is sequential or independent, process differently
        if isinstance(model, ConceptBottleneckModel):
            y_test_pred = model.predict(X_test.reshape(-1, orig_dims[0],
                                                orig_dims[1], orig_dims[2]))
            y_test_pred = np.argmax(y_test_pred, axis=1)
            acc = accuracy_score(y_test_pred, y_test)
            cm = confusion_matrix(y_test_pred, y_test)
        else:
            preds = model.predict(X_test.reshape(-1, orig_dims[0], orig_dims[1],
                                                orig_dims[2]))
            y_test_pred = preds[-1]
            pred = y_test_pred > 0.5
            acc = accuracy_score(pred, y_test)
            cm = confusion_matrix(pred, y_test)

    return acc, cm


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

    return lda

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
    out = layers.Dense(1, activation="sigmoid")(x)

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

def cbm_binary_classifier(dataset, training_mode, X_train, c_train, y_train, 
    X_valid, c_valid, y_valid, untrained_cto, path=None):
    """
    Train a concept bottleneck model with training procedure as specified by
    training_mode.

    :param dataset: one of Dataset in constants.py.
    :param training_mode: one of the ConceptBottleneckModel training procedure,
        as specified in constants.py.
    :param X_train, c_train, y_train: the training data (non-flatten).
    :param X_valid, c_valid, y_valid: the validation data.
    :param untrained_cto: untrained concept-to-output model (sklearn LR or DT by default)
    :param path: if specified, save model to the path.

    :return: ConceptBottleneckModel in shift_dimensionality_reductor.py. Please
        see the file for the class functions.
    """

    orig_dims = X_train.shape[1:]

    ## The input to concept and concept to output models.
    _, itc_model = multitask_model(dataset, X_train, c_train, X_valid, c_valid, train=False)
    cto_model = untrained_cto

    ## Trained accordingly
    # If independent, train the model separately
    if training_mode == ConceptBottleneckTraining.Independent:
        # Train the itc model
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        epochs = 1000
        batch_size = 128

        # Get the y
        y_train_internal = [c_train[:, i] for i in range(c_train.shape[1])]
        y_valid_internal = [c_valid[:, i] for i in range(c_valid.shape[1])]

        itc_model.fit(x=X_train, y=y_train_internal, 
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(X_valid, y_valid_internal),
                            callbacks=[lr_reducer, early_stopper])
        
        # Train the cto model
        cto_model.fit(c_train, y_train)

        cbm = ConceptBottleneckModel(itc_model, cto_model, dataset)

        if path:
            with open(path, "wb") as handle:
                pickle.dump(cbm, handle)
                print("Saving CBM successfully.")
        
        return cbm
    
    # If sequential, train the cto model using prediction result of the itc model
    elif training_mode == ConceptBottleneckTraining.Sequential:
        # Train the itc model
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        epochs = 1000
        batch_size = 128

        # Get the y
        y_train_internal = [c_train[:, i] for i in range(c_train.shape[1])]
        y_valid_internal = [c_valid[:, i] for i in range(c_valid.shape[1])]

        itc_model.fit(x=X_train, y=y_train_internal, 
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(X_valid, y_valid_internal),
                            callbacks=[lr_reducer, early_stopper])
        
        preds = itc_model.predict(X_train)
        concept_repr = np.zeros_like(c_train)
        for i in range(c_train.shape[1]):
            concept_repr[:, i] = np.argmax(preds[i], axis=1)
        
        cto_model.fit(concept_repr, y_train)

        cbm = ConceptBottleneckModel(itc_model, cto_model, dataset)

        if path:
            with open(path, "wb") as handle:
                pickle.dump(cbm, handle)
                print("Saving CBM successfully.")

        return cbm
    
    # Else joint training with lambda = # of concepts
    else:
        img_inputs = Input(shape=(orig_dims[0], orig_dims[1], orig_dims[2]))
        x = SharedCNNBlock(dataset)(img_inputs)

        # Concept head layers depending on dataset
        if dataset == Dataset.DSPRITES:
            concepts_name = ["color", "shape", "scale", "rotation", "x", "y"]
            concepts_size = np.array([1, 3, 6, 40, 32, 32])
        elif dataset == Dataset.SMALLNORB:
            concepts_name = ["category", "instance", "elevation", "azimuth", "lighting"]
            concepts_size = np.array([5, 10, 9, 18, 6])
        else:
            concepts_name = ["floor", "wall", "object", "scale", "shape", "orientation"]
            concepts_size = np.array([10, 10, 10, 8, 4, 15])

        outputs = []
        new_x = []
        for concept_name, concept_size in zip(concepts_name, concepts_size):
            head_layer = layers.Dense(concept_size, activation="softmax", 
                                            name=concept_name)(x)
            outputs.append(head_layer)
            new_x.append(head_layer)
        
        # Merge layers
        x = layers.concatenate(new_x)
        out = layers.Dense(1, activation="sigmoid")(x)
        outputs.append(out)

        model = tf.keras.Model(inputs=img_inputs, outputs=outputs)
        loss = [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) for i in concepts_name]
        loss.append(tf.keras.losses.BinaryCrossentropy(from_logits=False))
        loss_weights = [1. for i in concepts_name]
        loss_weights.append(len(concepts_name))

        # compile and train
        optimizer = tf.keras.optimizers.Adam(lr=1e-4, amsgrad=True)
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=["accuracy"], loss_weights=loss_weights
                    )
        
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        epochs = 1000
        batch_size = 128

        # Train model
        y_train_internal = [c_train[:, i] for i in range(c_train.shape[1])]
        y_train_internal.append(y_train)
        y_valid_internal = [c_valid[:, i] for i in range(c_valid.shape[1])]
        y_valid_internal.append(y_valid)

        model.fit(x=X_train, y=y_train_internal, 
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(X_valid, y_valid_internal),
                        callbacks=[lr_reducer, early_stopper])

        if path:
            model.save(path)

        return model


#-------------------------------------------------------------------------------
## Dataset construction

def generate_domain_classifier_data(X_train, y_train, 
        c_train, X_test, y_test, c_test, balanced=True):
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

    :return: X_train_new, y_train_new, c_train_new, 
        X_val_new, y_val_new, c_val_new, X_test_new, y_test_new, c_test_new.
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

    # Concept information
    c_clean_train = c_train[training_indices, :]
    c_clean_val = c_train[validation_indices, :]
    c_clean_test = c_train[test_indices, :]

    c_altered_train = c_test[training_indices, :]
    c_altered_val = c_test[validation_indices, :]
    c_altered_test = c_test[test_indices, :]

    ## Recombine datasets
    X_train_new = np.append(X_clean_train, X_altered_train, axis=0)
    y_train_new = np.zeros(len(X_train_new))
    y_train_new[len(X_clean_train):] = np.ones(len(X_altered_train))
    c_train_new = np.append(c_clean_train, c_altered_train, axis=0)

    X_test_new = np.append(X_clean_test, X_altered_test, axis=0)
    y_test_new = np.zeros(len(X_test_new))
    y_test_new[len(X_clean_test):] = np.ones(len(X_altered_test))
    c_test_new = np.append(c_clean_test, c_altered_test, axis=0)

    X_val_new = np.append(X_clean_val, X_altered_val, axis=0)
    y_val_new = np.zeros(len(X_val_new))
    y_val_new[len(X_clean_val):] = np.ones(len(X_altered_val))
    c_val_new = np.append(c_clean_val, c_altered_val, axis=0)

    ## Shuffle them
    X_train_new, y_train_new, c_train_new = unison_shuffled_copies(X_train_new, y_train_new, c_train_new)
    X_test_new, y_test_new, c_test_new = unison_shuffled_copies(X_test_new, y_test_new, c_test_new)
    X_val_new, y_val_new, c_val_new = unison_shuffled_copies(X_val_new, y_val_new, c_val_new)

    return X_train_new, y_train_new, c_train_new, X_val_new, y_val_new, c_val_new, X_test_new, y_test_new, c_test_new


#-------------------------------------------------------------------------------
## Helper functions 

def unison_shuffled_copies(a, b, c):
    """
    Used to shuffle a, b, c together.

    :param a, b, c: arrays of same length.

    :return: shuffled a, b, c
    """

    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]
