import numpy as np
import pandas as pd


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
    augmented = np.zeros((len(image_data)*5, 110, 110))
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
        i += 5
        j += 1

    return augmented, pd.DataFrame(augmented_labels)


def rotate_180(img):
    ''' Rotates image data 180 degrees,
    concatenates two np.rot90 statemenets. '''

    return np.rot90(np.rot90(img))


def rotate_270(img):
    ''' Rotates image data 270 degrees,
    concatenates three np.rot90 statemenets. '''

    return np.rot90(np.rot90(np.rot90(img)))
