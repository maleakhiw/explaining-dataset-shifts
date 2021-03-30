#-------------------------------------------------------------------------------
# SHIFT REDUCTOR
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains various dimensionality reduction methods
#  that we experimented with (see pipeline in our paper).
#-------------------------------------------------------------------------------

import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers, layers, Input
from sklearn.linear_model import LogisticRegression

from constants import *
from dsprites_utils import *


#-------------------------------------------------------------------------------
## Standard methods

def principal_components_analysis(X, n_components=None):
    """
    Fit a PCA dimensionality reductor.

    :param X: datasets to be fitted.
    :param n_components: number of components PCA.
    """

    # If number of components is not specified, calculate first
    if n_components is None:
        # Explain 80% variance of the original data
        pca = PCA(n_components=.8, svd_solver="full")
        pca.fit(X)
        n_components = pca.n_components_ 
    
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    return pca, n_components


def sparse_random_projection(X, n_components=None):
    """
    Fit a SRP dimensionality reductor.

    :param X: datasets to be fitted.
    :param n_components: number of components PCA.
    """

    # If number of components is not specified, calculate first
    # We want those that explain 80% of the original data
    if n_components is None:
        # Explain 80% variance of the original data
        pca = PCA(n_components=.8, svd_solver="full")
        pca.fit(X)
        n_components = pca.n_components_ 
    
    srp = SparseRandomProjection(n_components=n_components)
    srp.fit(X)
    
    return srp, n_components


#-------------------------------------------------------------------------------
## End-to-end neural network

def end_to_end_neural_network(num_classes, dataset):
    """
    End to end neural network that will be used as reduced representation for the
    BBSD-based methods.

    :param num_classes: the number of classes, for output layer.
    :param dataset: one of Dataset in constants.py.

    :return: the end-to-end neural network architecture.
    """

    if dataset == Dataset.DSPRITES:
        img_inputs = Input(shape=(64, 64, 1))

        # Shared layers
        x = SharedCNNBlock()(img_inputs)

        # Output layer
        out = layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=img_inputs, outputs=out)
    return model


#-------------------------------------------------------------------------------
## Concept bottleneck model (input-to-concept)

def multitask_model(dataset):
    """
    Multitask neural network that is tasked to predict all concepts jointly.

    :param dataset: one of Dataset in constants.py.
    """

    if dataset == Dataset.DSPRITES:
        img_inputs = Input(shape=(64, 64, 1))

        # Shared layers
        x = SharedCNNBlock()(img_inputs)

        # Task specific layer
        concepts_size = get_latent_sizes() # list describing number of possible concepts

        task_color = layers.Dense(concepts_size[0], activation="softmax", name="color")(x)
        task_shape = layers.Dense(concepts_size[1], activation="softmax", name="shape")(x)
        task_scale = layers.Dense(concepts_size[2], activation="softmax", name="scale")(x)
        task_rotation = layers.Dense(concepts_size[3], activation="softmax", name="rotation")(x)
        task_x = layers.Dense(concepts_size[4], activation="softmax", name="x")(x)
        task_y = layers.Dense(concepts_size[5], activation="softmax", name="y")(x)
        
        # Return model
        model = tf.keras.Model(inputs=img_inputs, outputs=[task_color, task_shape, task_scale, task_rotation, task_x, task_y])
    
    return model


#-------------------------------------------------------------------------------
## Concept bottleneck model (input-to-concept and concept-to-output wrapper)

class ConceptBottleneckModel:
    """
    Wrap input-to-concept and concept-to-output models into a single model.
    """

    def __init__(self, itc_model, cto_model, dataset):
        """
        Constructor.

        :param itc_model: the input to concept model.
        :param cto_model: the concept to output model.
        :param dataset: one of the Dataset constants.
        """

        self.itc_model = itc_model
        self.cto_model = cto_model
        self.dataset = dataset
    
    def predict(self, x):
        """
        Mimic how tensorflow predict function behaves.

        :param x: the images data.
        """

        # Input to concept prediction
        if self.dataset == Dataset.DSPRITES:
            preds = self.itc_model.predict(x)
            color_pred = np.argmax(preds[0], axis=1)
            shape_pred = np.argmax(preds[1], axis=1)
            scale_pred = np.argmax(preds[2], axis=1)
            rotation_pred = np.argmax(preds[3], axis=1)
            x_pred = np.argmax(preds[4], axis=1)
            y_pred = np.argmax(preds[5], axis=1)

            itc_preds = np.array([color_pred, shape_pred, scale_pred, 
                        rotation_pred, x_pred, y_pred]).T
        
            # Concept to output prediction
            return self.cto_model.predict_proba(itc_preds)


#-------------------------------------------------------------------------------
## Helper functions

class SharedCNNBlock(layers.Layer):
    """
    Standard shared convolutional block that is shared by the concept bottleneck
    model and end-to-end network.
    """

    def __init__(self, dataset):
        """
        Initialise convolutional blocks based on dataset name.

        :param dataset: one of Dataset in constants.py.
        """
        super(SharedCNNBlock, self).__init__()

        self.dataset = dataset

        if self.dataset == Dataset.DSPRITES:
            # Shared layers component
            self.conv1 = layers.Conv2D(64, (8, 8), strides=(2, 2), padding='same')
            self.do1 = layers.Dropout(0.3)

            self.conv2 = layers.Conv2D(128, (6, 6), strides=(2, 2), padding='valid')
            self.bn1 = layers.BatchNormalization()

            self.conv3 = layers.Conv2D(128, (5, 5), strides=(1, 1), padding='valid')
            self.pool1 = layers.MaxPooling2D((2, 2))
            self.do2 = layers.Dropout(0.3)

            self.flatten = layers.Flatten()
            self.dense1 = layers.Dense(128, activation="relu")
            self.do3 = layers.Dropout(0.4)
            self.dense2 = layers.Dense(64, activation="relu")
            self.do4 = layers.Dropout(0.2)
    
    def call(self, input):
        """
        Given an input, return outputs after passed to the shared layers.
        """

        if self.dataset == Dataset.DSPRITES:
            x = self.conv1(input)
            x = self.do1(x)
            x = self.conv2(x)
            x = self.bn1(x)
            x = self.conv3(x)
            x = self.pool1(x)
            x = self.do2(x)
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.do3(x)
            x = self.dense2(x)
            x = self.do4(x)

        return x