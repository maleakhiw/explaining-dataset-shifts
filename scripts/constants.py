#-------------------------------------------------------------------------------
# CONSTANTS
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains constants, including shift intensity levels,
#     shift type, dimensionality reduction method, etc.
#-------------------------------------------------------------------------------

from enum import Enum


#-------------------------------------------------------------------------------
## Constants

MAJORITY = "MAJORITY"



#-------------------------------------------------------------------------------
## Shift configurations

class ShiftIntensity(Enum):
    """
    Constants for shift intensity to be applied.
    """

    Small = 0
    Medium = 1
    Large = 2


class ShiftImageType(Enum):
    """
    Constants for shift image type to be applied.
    """

    Width = 0
    Height = 1
    Rotation = 2
    Shear = 3
    Zoom = 4
    Flip = 5
    All = 6


class ImageDataGeneratorConfig:
    """
    Configuration for ImageDataGeneratior.
    """

    # Rotation
    Rotation = {
        ShiftIntensity.Small: 10,
        ShiftIntensity.Medium: 40,
        ShiftIntensity.Large: 90
    }

    # Width shift (x-translation)
    Width = {
        ShiftIntensity.Small: 0.05,
        ShiftIntensity.Medium: 0.2,
        ShiftIntensity.Large: 0.4
    }

    # Height shift (y-translation)
    Height = {
        ShiftIntensity.Small: 0.05,
        ShiftIntensity.Medium: 0.2,
        ShiftIntensity.Large: 0.4
    }

    # Shear
    Shear = {
        ShiftIntensity.Small: 0.1,
        ShiftIntensity.Medium: 0.2,
        ShiftIntensity.Large: 0.3
    }

    # Zoom
    Zoom = {
        ShiftIntensity.Small: 0.1,
        ShiftIntensity.Medium: 0.2,
        ShiftIntensity.Large: 0.4
    }

    # Flip
    Flip = {
        ShiftIntensity.Small: (False, False),
        ShiftIntensity.Medium: (True, False),
        ShiftIntensity.Large: (True, True)
    }


#-------------------------------------------------------------------------------
## Statistical tests configuration

class OnedimensionalTest(Enum):
    KS = 0
    AD = 1


#-------------------------------------------------------------------------------
## Dataset configuration

class Dataset(Enum):
    """
    Constants representing the dataset.
    """

    DSPRITES = 0
    3DSHAPES = 1
    SMALLNORB = 2
