#-------------------------------------------------------------------------------
# SHIFT APPLICATOR
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains functions to apply various types of shifts.
#-------------------------------------------------------------------------------

import numpy as np
from copy import deepcopy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

from constants import *


#-------------------------------------------------------------------------------
## Shift applicator

def apply_gaussian_shift(X_te_orig, y_te_orig, shift_intensity, shift_prop):
    """
    Given a dataset (test), this function applies Gaussian shift of intensity,
    as specified in shift_intensity to a proportion of data, as indicated by the
    shift_prop.
    
    :param X_te_orig: the feature matrix which we will apply shifts.
    :param y_te_orig: the label array which we will apply shifts.
    :param shift_intensity: ShiftIntensity.Small, ShiftIntensity.Medium, 
        ShiftIntensity.Large. Describe the intensity of shift to apply to the 
        dataset.
    :param shift_prop: the proportion of the data affected by shift.

    :return X_te_1: shifted feature matrix.
    :return y_te_1: shifted label array.
    """

    X_te_1 = None
    y_te_1 = None

    # Gaussian noise shift on the features
    if shift_intensity == ShiftIntensity.Large:
        noise_amt = 100.0
    elif shift_intensity == ShiftIntensity.Medium:
        noise_amt = 10.0
    else:
        noise_amt = 1.0

    normalization = 255.0 # normalization constant for image data

    X_te_1, _ = gaussian_noise_subset(X_te_orig, noise_amt, 
        normalization=normalization, delta_total=shift_prop)
    y_te_1 = y_te_orig.copy()
    
    return (X_te_1, y_te_1)


def apply_ko_shift(X_te_orig, y_te_orig, shift_intensity, cl=MAJORITY):
    """
    Given a dataset (test), this function applies the knockout shift, removing 
    data from a particular class.
    
    :param X_te_orig: the feature matrix which we will apply shifts.
    :param y_te_orig: the label array which we will apply shifts.
    :param shift_intensity: ShiftIntensity.Small, ShiftIntensity.Medium, 
        ShiftIntensity.Large. Describe the intensity of shift to apply to the 
        dataset.
    :param cl: the class which we will remove. For the class with most element, 
        we specify MAJORITY.

    :return X_te_1: shifted feature matrix.
    :return y_te_1: shifted label array.
    """

    X_te_1 = None
    y_te_1 = None

    # Knockout shift, creating class imbalance on the dataset
    if shift_intensity == ShiftIntensity.Large:
        prop = 1.0
    elif shift_intensity == ShiftIntensity.Medium:
        prop = 0.5
    else:
        prop = 0.1
    
     # If want to delete majority
    if cl == MAJORITY:
        # Find majority
        c = Counter(y_te_orig)
        cl, _ = c.most_common()[0] 
    
    X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, cl, prop)

    return (X_te_1, y_te_1)


def apply_concept_shift(X_te_orig, y_te_orig, c_te_orig, concept_idx, 
    shift_intensity, cl=MAJORITY):
    """
    Given a dataset (test), this function applies shift to a particular concept,
    creating imbalance in the specified concept.

    :param X_te_orig: the feature matrix which we will apply shifts.
    :param y_te_orig: the label array which we will apply shifts.
    :param c_te_orig: the concept matrix which we will apply shifts.
    :param concept_idx: index of the concept to be shifted
    :param shift_intensity: ShiftIntensity.Small, ShiftIntensity.Medium, 
        ShiftIntensity.Large. Describe the intensity of shift to apply to the 
        dataset.
    :param cl: the class which we will remove. For the class with most element, 
        we specify MAJORITY.

    :return X_te_1: shifted feature matrix.
    :return y_te_1: shifted label array.
    :return c_te_1: shifted concept matrix.
    """

    X_te_1 = None
    c_te_1 = None
    y_te_1 = None
    c_te_1 = deepcopy(c_te_orig)
    c_te_orig = c_te_orig[:, concept_idx]

    # Shift intensities settings
    if shift_intensity == ShiftIntensity.Large:
        prop = 1.0
    elif shift_intensity == ShiftIntensity.Medium:
        prop = 0.5
    else:
        prop = 0.1
    
    # If want to delete majority
    if cl == MAJORITY:
        # Find majority
        c = Counter(c_te_orig)
        cl, _ = c.most_common()[0] 
    
    X_te_1, del_indices = knockout_shift(X_te_orig, c_te_orig, cl, 
        prop, return_indices=True)
    not_del_indices = [i for i in range(len(y_te_orig)) if i not in del_indices]
    c_te_1 = c_te_1[not_del_indices, :]
    y_te_1 = np.delete(y_te_orig, del_indices, axis=0)

    return (X_te_1, y_te_1, c_te_1)


def apply_img_shift(X_te_orig, y_te_orig, orig_dims, shift_intensity, 
    shift_prop, shift_types=ShiftImageType.All):
    """
    Given a dataset (test), this function applies various image shifts to it
    
    :param X_te_orig: the feature matrix which we will apply shifts.
    :param y_te_orig: the label array which we will apply shifts.
    :param orig_dims: original dimensions of the image (width, height, channel)
    :param shift_intensity: ShiftIntensity.Small, ShiftIntensity.Medium, 
        ShiftIntensity.Large. Describe the intensity of shift to apply to the 
        dataset.
    :param shift_prop: proportion of the data shifted, default value = 0.1, 0.5, 1.0.
    :param shift_types: list indicating shifts that we want to apply. 
        Possible shift types includes:
        - ShiftImageType.Width: for image translation in the x-direction.
        - ShiftImageType.Height: for image translation in the y-direction.
        - ShiftImageType.Rotation: rotation.
        - ShiftImageType.Shear: shear.
        - ShiftImageType.Zoom: zoom along the x and y directions.
        - ShiftImageType.Flip: flip in the x and y direction.
        - ShiftImageType.All: all combination of image shifts above.
    
    :return X_te_1: shifted feature matrix.
    :return y_te_1: shifted label array.
    """

    X_te_1 = None
    y_te_1 = None

    ## No specific type of image shifts given
    if shift_type == ShiftImageType.All :
        rotation_range = ImageDataGeneratorConfig.Rotation[shift_intensity]
        width_shift_range = ImageDataGeneratorConfig.Width[shift_intensity]
        height_shift_range = ImageDataGeneratorConfig.Height[shift_intensity]
        shear_range = ImageDataGeneratorConfig.Shear[shift_intensity]
        zoom_range = ImageDataGeneratorConfig.Zoom[shift_intensity]
        horizontal_flip = ImageDataGeneratorConfig.Flip[shift_intensity][0]
        vertical_flip = ImageDataGeneratorConfig.Flip[shift_intensity][1]
    
    ## Else, select parameters accordingly
    else:
        all_shifts = [
            ShiftImageType.Width, ShiftImageType.Height, ShiftImageType.Rotation, 
            ShiftImageType.Shear, ShiftImageType.Zoom, ShiftImageType.Flip
        ]

        for shift_type in shift_types:
            if shift_type == ShiftImageType.Width:
                width_shift_range = ImageDataGeneratorConfig.Width[shift_intensity]
            if shift_type == ShiftImageType.Height:
                height_shift_range = ImageDataGeneratorConfig.Height[shift_intensity]
            if shift_type == ShiftImageType.Rotation:
                rotation_range = ImageDataGeneratorConfig.Rotation[shift_intensity]
            if shift_type == ShiftImageType.Shear:
                shear_range = ImageDataGeneratorConfig.Shear[shift_intensity]
            if shift_type == ShiftImageType.Zoom:
                zoom_range = ImageDataGeneratorConfig.Zoom[shift_intensity]
            if shift_type == ShiftImageType.Flip:
                horizontal_flip = ImageDataGeneratorConfig.Flip[shift_intensity][0]
                vertical_flip = ImageDataGeneratorConfig.Flip[shift_intensity][1]
            
            all_shifts.remove(shift_type)
        
        # For non-included, use default value without augmentation
        for shift_type in all_shifts:
            if shift_type == ShiftImageType.Width:
                width_shift_range = 0
            if shift_type == ShiftImageType.Height:
                height_shift_range = 0
            if shift_type == ShiftImageType.Rotation:
                rotation_range = 0
            if shift_type == ShiftImageType.Shear:
                shear_range = 0
            if shift_type == ShiftImageType.Zoom:
                zoom_range = 0
            if shift_type == ShiftImageType.Flip:
                horizontal_flip = False
                vertical_flip = False

    # Apply shift using the parameters
    X_te_1, _ = image_generator(X_te_orig, 
                                orig_dims, 
                                rotation_range, 
                                width_shift_range,
                                height_shift_range, 
                                shear_range, 
                                zoom_range, 
                                horizontal_flip, 
                                vertical_flip, 
                                delta=shift_prop)
    y_te_1 = y_te_orig.copy()
    
    return (X_te_1, y_te_1)


#-------------------------------------------------------------------------------
## Helper

def gaussian_noise_subset(x, noise_amt, normalization=1.0, delta_total=1.0, 
                    clip=True):
    """
    Apply gaussian noise to set of images x.

    :param x: the feature matrix.
    :param noise_amt: the amount of Gaussian noise applied.
    :param normalization: max range of values, for image it would be 255.
    :param delta_total: proportion of data which we randomly apply noise.
    :param clip: whether to clip the new result between 0 and 1.

    :return x: the data after being corrupted by the Gaussian noise.
    :return indices: the indices of data instances which we applied noise.
    """
    
    # Indices of images where we apply noise (random indices)
    indices = np.random.choice(x.shape[0], np.ceil(x.shape[0] * delta_total), replace=False)
    x_mod = x[indices, :] # images
    
    # Create noise of appropriate size for all pixels (or other data structures)
    noise = np.random.normal(0, noise_amt / normalization, 
        (x_mod.shape[0], x_mod.shape[1]))
    
    # Clip X
    if clip:
        x_mod = np.clip(x_mod + noise, 0., 1.)
    else:
        x_mod = x_mod + noise
    
    # Return noisy X
    x[indices, :] = x_mod
    return x, indices


def knockout_shift(x, y, cl, delta, return_indices=False):
    """
    The knockout shift remove instances from a class in order to create class imbalance.
    
    :param x: the feature matrix.
    :param y: the label array.
    :param cl: the class label where we will remove its instances to create imbalance.
    :param delta: proportion of class to be removed.
    :param return_indices: whether to return deleted indices.

    :return x: the x after we removed instances of class cl.
    :return y: the y after we removed instances of class cl.
    :return del_indices: if needed, return the indices of deleted elements.
    """

    # Prevent error, prevent deletion if list contains only a single element
    if len(y) == 1 or len(x) == 1:
        # Lazy evaluation in edge case
        if not return_indices:
            return x, y
        else:
            return x, []
    
    # Indices to be deleted
    del_indices = np.where(y == cl)[0]
    until_index = np.ceil(delta * len(del_indices))
    
    # Prevent error (delete too much)
    # You cannot delete everything.
    if until_index == len(y):
        until_index = until_index - 1

    del_indices = del_indices[:until_index]
    
    # Delete instances of del_indices
    x = np.delete(x, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)
    
    # Return reduced data
    if not return_indices:
        return x, y
    else:
        return x, del_indices