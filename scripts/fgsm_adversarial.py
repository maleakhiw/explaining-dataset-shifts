#-------------------------------------------------------------------------------
# ADVERSARIAL (FGSM) SAMPLES
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains code used to generate FGSM adversarial samples
#    for various datasets and models. We load the pretrained models that we have
#    which include end-to-end neural network model for various tasks and datasets.
#-------------------------------------------------------------------------------

import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import numpy as np


#-------------------------------------------------------------------------------
## Adversarial generator

def generate_adversarials(X_test, y_test, model, dataset_name, plot_sample,
                        epsilon=0.1, n_plots=10):
    """
    Generate adversarial samples (using Goodfellow et al's FGSM) and store the 
    results in the GitHub repository. Under the data folder.

    :param X_test: the datasets in which we will attack using FGSM (we assume the 
        X_test is non-flatten). We assume that image has been normalised beforehand.
    :param y_test: the y label of the X_test.
    :param model: the trained model which we will fool.
    :param dataset_name: dataset name (string e.g., dsprites, 3dshapes, etc)
    :param plot_sample: flag indicating whether to plot adversarial samples.
    :param epsilon: used as FGSM parameter, determine the degree to which the original
        image is affected by the adversarial pattern.
    :param n_plots: number of adversarial plots to be saved to be drawn.
    """

    # Sanity check to verify whether image has been normalised before.
    maximum_pixel_value = np.max(X_test)
    if maximum_pixel_value > 1.0:
        X_test = X_test / 255.0

    orig_dims = X_test.shape[1:]

    # Turn all test set samples into adversarial samples
    for i in tqdm(range(len(X_test))):
        # Create adversarial pattern
        perturbations = adversarial_pattern(X_test[i].reshape(1, orig_dims[0], orig_dims[1], orig_dims[2]),
                                        model, y_test[i]).numpy()
        adversarial_sample = X_test[i] + perturbations * epsilon

        # Draw adversarial sample if required.
        if plot_sample and i < n_plots:
            print(f"Prediction: {model.predict(adversarial_sample.reshape((1, orig_dims[0], orig_dims[1], orig_dims[2]))).argmax()} \
                | Truth: {model.predict(X_test[i].reshape((1, orig_dims[0], orig_dims[1], orig_dims[2]))).argmax()}")

            # Plot
            # Channel = 1
            if orig_dims[2] == 1:
                fig = plt.imshow(adversarial_sample.reshape(orig_dims[0], orig_dims[1]), cmap="gray")
            else:
                fig = plt.imshow(adversarial_sample.reshape(orig_dims[0], orig_dims[1], orig_dims[2]))
            
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.show()

        # Successful adversarial samples are written back into original matrix.
        X_test[i] = np.reshape(adversarial_sample, np.prod(orig_dims)) # flatten version

    # Save results
    np.save(f"../data/adversarial_samples/X_adversarial_{dataset_name}.npy", X_test)
    np.save(f"../data/adversarial_samples/y_adversarial_{dataset_name}.npy", y_test)

#-------------------------------------------------------------------------------
## Helper functions

def adversarial_pattern(image, model, label):
    """
    Given image and true label, we generate the perturbation pattern
    using Goodfellow's FGSM.

    :param image: the image where we will find the pattern.
    :param model: the pretrained model which we will fool.
    :param label: the corresponding image label (as integer).

    :return: array of adversarial pattern of the same size as original image.
    """

    # Convert image to tensor array
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)

    # Get gradient
    gradient = tape.gradient(loss, image) # gradient wrt input image (see paper)
    signed_grad = tf.sign(gradient)

    return signed_grad
