#-------------------------------------------------------------------------------
# ADVERSARIAL SAMPLES
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains code used to generate FGSM adversarial samples
#    for various datasets and models. We load the pretrained models that we have
#    which include end-to-end model for various tasks and datasets, and the 
#    concept bottleneck models.
# Credit: This file is forked from 
#  https://github.com/bethgelab/foolbox and https://github.com/steverab/failing-loudly.
#-------------------------------------------------------------------------------

import os 
import matplotlib.pyplot as plt
from tqdm import tqdm

from foolbox.models import KerasModel
from foolbox.attacks import FGSM

from tensorflow.keras.models import load_model

from constants import *


#-------------------------------------------------------------------------------
## Main function

def generate_adversarial_samples(dataset, X_test, y_test, model, filename, plot_sample=True, 
                            max_epsilon=1.0, n_plots=100):
    """
    Generate adversarial samples (using Goodfellow et al's FGSM) and store the 
    results in the GitHub repository. Under the data folder.

    :param dataset: one of Dataset objects in the constants.py.
    :param X_test: the datasets in which we will attack using FGSM (we assume the 
        X_test is non-flatten).
    :param y_test: the y label of the X_test.
    :param model: the trained model which we will fool.
    :param filename: the filename to store the adversarial samples.
    :param plot_sample: flag indicating whether to plot adversarial samples.
    :param max_epsilon: used as FGSM parameter, determine step size (see foolbox documentation)
    :param n_plots: number of adversarial plots to be saved to be drawn.
    """

    # Normalise datapoints
    X_test = X_test.astype("float32") / 255.0
    orig_dims = X_test.shape[1:]

    # Create foolbox model from the tensorflow 2 keras model.
    foolbox_model = KerasModel(model, (0, 1))
    attack = FGSM(foolbox_model)

    # Turn all test set samples into adversarial samples
    for i in tqdm(range(len(X_test))):
        # Create adversarial sample
        adv_sample = attack(X_test[i], label=y_test[i], max_epsilon=max_epsilon)

        if adv_sample is not None:
            # Check whether their label is different and how the sample looks like
            if plot_sample and i < n_plots:
                orig_pred = model.predict(X_test[i])
                orig_pred = np.asscalar(np.argmax(orig_pred, axis=1))
                print(f"Original prediction: {orig_pred}")
                pred = model.predict(adv_sample.reshape(1,
                                                        orig_dims[0],
                                                        orig_dims[1],
                                                        orig_dims[2]))
                pred = np.asscalar(np.argmax(pred, axis=1))
                print(f"Adversarial prediction: {pred}")

                # Plot sample
                fig = plt.imshow(adv_sample.reshape(orig_dims[0], orig_dims[1], 
                                            cmap="gray")
                plt.axis("off")
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.show()

            # Successful adversarial samples are written back into original matrix.
            X_test[i] = np.reshape(adv_sample, np.prod(orig_dims)) # flatten version

    # Save results
    np.save("../data/adversarial_samples/X_adversarial_dsprites.npy", X_test)
    np.save("../data/adversarial_samples/y_adversarial_dsprites.npy", y_test)


#-------------------------------------------------------------------------------