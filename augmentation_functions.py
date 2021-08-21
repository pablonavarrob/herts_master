import pandas as pd
import numpy as np
import random
import tensorflow as tf
from PIL import Image, ImageEnhance

def contrast_randomize(img, PIL_output=False):
    ''' Randomly increases/decreases the contrast of the input image. '''

    img = array_shape_check(img)
    # Convert to PIL format
    img = Image.fromarray(img.astype(np.uint8))
    # Increase contrast using random integer
    rand_contrast = np.random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(rand_contrast)
    # Return an array of the same shape or a PIL image if used concat
    if PIL_output:
        return img
    else:
        return np.asarray(img, dtype='uint8')

def exposure_randomize(img, PIL_input):
    ''' Randomly increases/decreases the exposure of the input image. '''

    if PIL_input:
        pass
    else:
        # Convert to PIL format
        img = array_shape_check(img)
        img = Image.fromarray(img.astype(np.uint8))

    # Increase contrast using random integer
    rand_exp = np.random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(rand_exp)

    # Return an array of the same shape or a PIL image if used concat
    return np.asarray(img, dtype='uint8')

def random_flip(img):
    ''' Straightforward randomization of the image flip '''

    int = random.randint(2, 4)
    img = array_shape_check(img)

    # if int == 1:
    #     # No change
    #     print('No change')
    #     return img

    if int == 2:
        # Up-down flip
        print('Up-down flip')
        return np.flipud(img)

    elif int == 3:
        # Left-right flip
        print('Left-right flip')
        return np.fliplr(img)

    elif int == 4:
        # Up-down and left-right
        print('Left-right and up-down')
        return np.fliplr(np.flipud(img))


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

def augmentation(image_data, image_label):
    ''' For each one of the input samples that does have a volcano,
    we create different versions, that is, randomized brighness and contrast
    as well as flips up-down and left-right.

    This function thus creates a larger set of images containing volcanoes,
    however there will still be an imbalace in the subclasses.

    The output comprises both the non-augmneted images and their augmented
    counterparts - only one augmentation per image. '''

    # image_data = image_data.to_numpy().reshape(len(image_data), 110, 110)
    augmented = np.zeros((len(image_data), 110, 110))
    labels = []
    i = 0
    j = 0
    for img in image_data:
        augmented[i] = (
            random_flip(
                exposure_randomize(
                    contrast_randomize(img, PIL_output=True),
                    PIL_input=True
                    )
                )
            )
        labels.append(image_label.iloc[i])
        # augmented[i+1] = img
        # labels.append(image_label.iloc[j])
        # # Add index
        # print(i, j)
        i += 1
        # j += 1

    return augmented, labels
