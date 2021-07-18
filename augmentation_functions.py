import numpy as np
import pandas as pd


def augmentation(image_data):
    ''' For each one of the input samples that does have a volcano,
    we create different versions, that is: rotated 90, 180 and 270
    degrees plus on that is mirrored.

    Labels are necessary as well, as only those images that do
    contain volcanoes will be agumented to compensate for the
    imbalanced classes '''

    augmented = np.zeros((len(image_data)*5, 110, 110))
    i = 0
    for img in image_data:
        augmented[i+0] = np.rot90(img)
        augmented[i+1] = rotate_180(img)
        augmented[i+2] = rotate_270(img)
        augmented[i+3] = np.flipud(img)  # Flip upside down
        augmented[i+4] = np.fliplr(img)  # Mirror left to right
        i += 5

    return augmented


def rotate_180(img):
    ''' Rotates image data 180 degrees,
    concatenates two np.rot90 statemenets. '''

    return np.rot90(np.rot90(img))


def rotate_270(img):
    ''' Rotates image data 270 degrees,
    concatenates three np.rot90 statemenets. '''

    return np.rot90(np.rot90(np.rot90(img)))
