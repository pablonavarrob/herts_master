import numpy as np
import pandas as pd
from tensorflow.image import random_contrast, random_brightness
import tensorflow as tf

def augmentation(image_data, label_data):
    ''' For each one of the input samples that does have a volcano,
    we create different versions, that is: rotated 90, 180 and 270
    degrees plus on that is mirrored.

    Labels are necessary as well, as only those images that do
    contain volcanoes will be agumented to compensate for the
    imbalanced classes '''

    def append_labels(idx, augmentation_type):
        augmented_labels.append({
            "Volcano?": label_data.iloc[idx]["Volcano?"],
            "Type": label_data.iloc[idx]["Type"],
            "Radius": label_data.iloc[idx]["Radius"],
            "Number Volcanoes": label_data.iloc[idx]["Number Volcanoes"],
            "Augmentation type": augmentation_type
        })

    augmented_labels = []  # Create empty list, append dictionaries
    augmented = np.zeros((len(image_data)*11, 110, 110))
    i = 0
    j = 0
    for img in image_data:

        # Data augmentation loop
        augmented[i+0] = np.rot90(img)
        append_labels(j, "rot90")
        augmented[i+1] = rotate_180(img)
        append_labels(j, "rot180")
        augmented[i+2] = rotate_270(img)
        append_labels(j, "rot270")
        augmented[i+3] = np.flipud(img)  # Flip upside down
        append_labels(j, "flipud")
        augmented[i+4] = np.fliplr(img)  # Mirror left to right
        append_labels(j, "fliplr")
        augmented[i+5] = contrast_randomize(img)  # Contrast
        append_labels(j, "contrast")
        augmented[i+6] = exposure_randomize(img)  # Exposure
        append_labels(j, "exposure")
        augmented[i+7] = np.fliplr(contrast_randomize(img))  # Contrast + fliplr
        append_labels(j, "contrast_fliplr")
        augmented[i+8] = np.fliplr(exposure_randomize(img))  # Exposure + fliplr
        append_labels(j, "exposure_fliplr")
        augmented[i+9] = np.flipud(contrast_randomize(img))  # Contrast + flipud
        append_labels(j, "contrast_flipud")
        augmented[i+10] = np.flipud(exposure_randomize(img))  # Exposure + flipud
        append_labels(j, "exposure_flipud")
        i += 11
        j += 1

    return augmented, pd.DataFrame(augmented_labels)


def shaper_tensor_check(img):
    ''' Checks whether the shape of the input image is accepted by tensorflow
    by added the third dimension. Used for the brightness and contrast
    augmentation. Input a numpy array. '''

    if img.shape != (110, 110, 1):
        img = img.reshape(110, 110, 1)
    else:
        pass

    return img


def array_shape_check(img):

    if img.shape != (110, 110):
        img = img.reshape(110, 110)
    else:
        pass

    return img


def rotate_180(img):
    ''' Rotates image data 180 degrees,
    concatenates two np.rot90 statemenets. '''

    img = array_shape_check(img)

    return np.rot90(np.rot90(img))


def rotate_270(img):
    ''' Rotates image data 270 degrees,
    concatenates three np.rot90 statemenets. '''

    img = array_shape_check(img)

    return np.rot90(np.rot90(np.rot90(img)))


def contrast_randomize(im

def exposure_randomize(img):
    ''' Randomly increases/decreases the exposure of the input image. '''

    img = shaper_tensor_check(img)

    return tf.reshape(random_brightness(img, 0.85), (110, 110)).numpy()
