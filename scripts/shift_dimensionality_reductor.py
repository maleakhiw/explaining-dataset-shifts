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
from tensorflow.keras import optimizers, layers, Input, Model
from sklearn.linear_model import LogisticRegression

from constants import *


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

def end_to_end_neural_network(num_classes, dataset, 
                            X_train, y_train, X_valid, y_valid, save_path=None):
    """
    End to end neural network that will be used as reduced representation for the
    BBSD-based methods.

    :param num_classes: the number of classes, for output layer.
    :param dataset: one of Dataset in constants.py.
    :param X_train, y_train, X_valid, y_valid: data used to train the neural network.
    :param save_path: if specified, save the model after training.

    :return: the end-to-end neural network architecture.
    """

    if dataset in {Dataset.SMALLNORB, Dataset.DSPRITES}:
        img_inputs = Input(shape=(64, 64, 1))
    else:
        img_inputs = Input(shape=(64, 64, 3)) # for 3dshapes

    # Shared layers
    x = SharedCNNBlock(dataset)(img_inputs)

    # Output layer
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=img_inputs, outputs=out)

    # Compile and train model
    optimizer = optimizers.Adam(lr=1e-4, amsgrad=True)
    epochs = 200
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    batch_size = 64

    optimizer = tf.keras.optimizers.Adam(lr=1e-4, amsgrad=True)
    model.compile(loss="sparse_categorical_crossentropy",
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


#-------------------------------------------------------------------------------
## Concept bottleneck model (input-to-concept)

def multitask_model(dataset, X_train, c_train, X_valid, c_valid, save_path=None,
            concepts_size=None):
    """
    Multitask neural network that is tasked to predict all concepts jointly.

    :param dataset: one of Dataset in constants.py.
    :param X_train, c_train, X_valid, c_valid: training and validation data (for early stopper).
    :param save_path: if path is specified, we will save the model, otherwise note.
    :param concepts_size: if specified, the concepts size is used (array of possible values for each concept).
    """

    ## dSprites
    if dataset == Dataset.DSPRITES:
        img_inputs = Input(shape=(64, 64, 1))

        # Shared layers
        x = SharedCNNBlock(dataset)(img_inputs)

        # Task specific layer
        if concepts_size is None:
            concepts_size = np.array([1, 3, 6, 40, 32, 32]) # list describing number of possible concepts

        task_color = layers.Dense(concepts_size[0], activation="softmax", name="color")(x)
        task_shape = layers.Dense(concepts_size[1], activation="softmax", name="shape")(x)
        task_scale = layers.Dense(concepts_size[2], activation="softmax", name="scale")(x)
        task_rotation = layers.Dense(concepts_size[3], activation="softmax", name="rotation")(x)
        task_x = layers.Dense(concepts_size[4], activation="softmax", name="x")(x)
        task_y = layers.Dense(concepts_size[5], activation="softmax", name="y")(x)
        
        model = tf.keras.Model(inputs=img_inputs, outputs=[task_color, task_shape, task_scale, task_rotation, task_x, task_y])
        
        # Compile and train model
        optimizer = tf.keras.optimizers.Adam(lr=1e-4, amsgrad=True)
        model.compile(optimizer=optimizer,
                    loss=[
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                    ], metrics=["accuracy"])
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        epochs = 200

        histories = model.fit(x=X_train, y=[c_train[:, 0], c_train[:, 1], c_train[:, 2],
                        c_train[:, 3], c_train[:, 4], c_train[:, 5]], 
                        epochs=epochs, batch_size=128,
                        validation_data=(X_valid, [c_valid[:, 0], c_valid[:, 1], 
                        c_valid[:, 2], c_valid[:, 3], c_valid[:, 4], c_valid[:, 5]]),
                        callbacks=[lr_reducer, early_stopper])

    ## smallNORB
    elif dataset == Dataset.SMALLNORB:
        img_inputs = Input(shape=(64, 64, 1))

        # Shared layers
        x = SharedCNNBlock(dataset)(img_inputs)

        # Task specific layer
        if concepts_size is None:
            concepts_size = np.array([ 5, 10,  9, 18,  6]) # list describing number of possible concepts

        task_category = layers.Dense(concepts_size[0], activation="softmax", name="category")(x)
        task_instance = layers.Dense(concepts_size[1], activation="softmax", name="instance")(x)
        task_elevation = layers.Dense(concepts_size[2], activation="softmax", name="elevation")(x)
        task_azimuth = layers.Dense(concepts_size[3], activation="softmax", name="azimuth")(x)
        task_lighting = layers.Dense(concepts_size[4], activation="softmax", name="lighting")(x)
        
        model = tf.keras.Model(inputs=img_inputs, outputs=[task_category, task_instance, 
                                task_elevation, task_azimuth, task_lighting])
        
        # Compile and train model
        optimizer = tf.keras.optimizers.Adam(lr=1e-4, amsgrad=True)
        model.compile(optimizer=optimizer,
                    loss=[
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    ], metrics=["accuracy"])
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        epochs = 200
        batch_size = 128

        histories = model.fit(x=X_train, y=[c_train[:, 0], c_train[:, 1], c_train[:, 2],
                        c_train[:, 3], c_train[:, 4]], 
                        epochs=epochs, batch_size=128,
                        validation_data=(X_valid, [c_valid[:, 0], c_valid[:, 1], 
                        c_valid[:, 2], c_valid[:, 3], c_valid[:, 4]]),
                        callbacks=[lr_reducer, early_stopper])
    
    ## 3dshapes
    else:
        img_inputs = Input(shape=(64, 64, 3))

        # Shared layers
        x = SharedCNNBlock(dataset)(img_inputs)

        # Task specific layer
        if concepts_size is None:
            concepts_size = np.array([10, 10,  10, 8, 4, 15]) # list describing number of possible concepts

        task_floor = layers.Dense(concepts_size[0], activation="softmax", name="floor")(x)
        task_wall = layers.Dense(concepts_size[1], activation="softmax", name="wall")(x)
        task_object = layers.Dense(concepts_size[2], activation="softmax", name="object")(x)
        task_scale = layers.Dense(concepts_size[3], activation="softmax", name="scale")(x)
        task_shape = layers.Dense(concepts_size[4], activation="softmax", name="shape")(x)
        task_orientation = layers.Dense(concepts_size[5], activation="softmax", name="orientation")(x)
        
        model = tf.keras.Model(inputs=img_inputs, outputs=[task_floor, task_wall, 
                                task_object, task_scale, task_shape, task_orientation])
        
        # Compile and train model
        optimizer = tf.keras.optimizers.Adam(lr=1e-4, amsgrad=True)
        model.compile(optimizer=optimizer,
                    loss=[
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    ], metrics=["accuracy"])
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        epochs = 200
        batch_size = 128

        histories = model.fit(x=X_train, y=[c_train[:, 0], c_train[:, 1], c_train[:, 2],
                        c_train[:, 3], c_train[:, 4], c_train[:, 5]], 
                        epochs=epochs, batch_size=128,
                        validation_data=(X_valid, [c_valid[:, 0], c_valid[:, 1], 
                        c_valid[:, 2], c_valid[:, 3], c_valid[:, 4], c_valid[:, 5]]),
                        callbacks=[lr_reducer, early_stopper])

    # Save if specified
    if save_path:
        model.save(save_path)

    return histories, model


#-------------------------------------------------------------------------------
## Autoencoders (untrained and trained)

def autoencoder(dataset, X_train, X_val, orig_dims, train=True):
    """
    Build autoencoder to reduce dimensionality representation. The autoencoder
    can be trained or untrained (e.g., using random weights).

    :param dataset: one of Dataset values in constants.py.
    :param X_train: the training dataset.
    :param X_valid: used for validating reconstruction loss.
    :param orig_dims: the original dimensions of the image.
    :param train: indicate whether to use trained or untrained autoencoder.

    :return: encoder network and the full autoencoder
    """

    X = X_train.reshape(-1, orig_dims[0], orig_dims[1], orig_dims[2])

    input_img = Input(shape=orig_dims)

    # Define AE architecture
    if dataset in {Dataset.SMALLNORB, Dataset.DSPRITES}:
        x = layers.Conv2D(64, (3, 3), padding='same')(input_img)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(16, (3, 3), padding='same')(encoded)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(1, (3, 3), padding='same')(x)
        decoded = layers.Activation('sigmoid')(x)
    ## 3dshapes
    else:
        x = layers.Conv2D(64, (3, 3), padding='same')(input_img)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(16, (3, 3), padding='same')(encoded)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(3, (3, 3), padding='same')(x)
        decoded = layers.Activation('sigmoid')(x)

    # Construct both an encoding model and a full encoding-decoding model. The first one will be used for mere
    # dimensionality reduction, while the second one is needed for training.
    encoder = Model(input_img, encoded)
    autoenc = Model(input_img, decoded)

    autoenc.compile(optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9), loss='binary_crossentropy')

    if train:
        autoenc.fit(X.reshape(len(X), orig_dims[0], orig_dims[1], orig_dims[2]), 
                    X.reshape(len(X), orig_dims[0], orig_dims[1], orig_dims[2]),
                    epochs=100,
                    batch_size=128,
                    validation_data=(X_val.reshape(len(X_val), orig_dims[0], orig_dims[1], orig_dims[2]), 
                                     X_val.reshape(len(X_val), orig_dims[0], orig_dims[1], orig_dims[2])),
                    shuffle=True)
                    
    return encoder, autoenc


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
        self.cto_model = cto_model # we assume that the concept-to-output model has predict_proba
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
        
        
        elif self.dataset == Dataset.SMALLNORB:
            preds = self.itc_model.predict(x)
            category_pred = np.argmax(preds[0], axis=1)
            instance_pred = np.argmax(preds[1], axis=1)
            elevation_pred = np.argmax(preds[2], axis=1)
            azimuth_pred = np.argmax(preds[3], axis=1)
            lighting_pred = np.argmax(preds[4], axis=1)

            itc_preds = np.array([category_pred, instance_pred, elevation_pred,
                        azimuth_pred, lighting_pred]).T
        
        else:
            preds = self.itc_model.predict(x)
            floor_pred = np.argmax(preds[0], axis=1)
            wall_pred = np.argmax(preds[1], axis=1)
            object_pred = np.argmax(preds[2], axis=1)
            scale_pred = np.argmax(preds[3], axis=1)
            shape_pred = np.argmax(preds[4], axis=1)
            orientation_pred = np.argmax(preds[5], axis=1)

            itc_preds = np.array([floor_pred, wall_pred, object_pred,
                        scale_pred, shape_pred, orientation_pred]).T
        
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

        # Just add an elif or else for other datasets if you with to use other
        # CNN architectures.
        if self.dataset in {Dataset.SMALLNORB, Dataset.DSPRITES, Dataset.SHAPES3D}:
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

        if self.dataset in {Dataset.DSPRITES, Dataset.SMALLNORB, Dataset.SHAPES3D}:
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