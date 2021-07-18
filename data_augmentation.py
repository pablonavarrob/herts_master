import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import (load_volcano_data,
            rotate_180, rotate_270, augmentation)

# Load the data
train_images, train_labels, test_images, test_labels = load_volcano_data()

# Remix the data for the augmentation
total_img = pd.concat([train_images, test_images])
total_lbl = pd.concat([train_labels, test_labels])

# ------------------------------------------------------------------------------
## SELECT THE IMAGES CONTAINING VOLCANOES FOR THE AUGMENTATION PROCEDURE ##

volcano_t1_img_data = (
    total_img
    .loc[(total_lbl['Volcano?'] == 1) &
        (total_lbl['Type'] == 1)]
)
volcano_t1_img_data = (
    volcano_t1_img_data
    .to_numpy()
    .reshape(len(volcano_t1_img_data), 110, 110)
)

volcano_t2_img_data = (
    total_img
    .loc[(total_lbl['Volcano?'] == 1) &
        (total_lbl['Type'] == 2)]
)
volcano_t2_img_data = (
    volcano_t2_img_data
    .to_numpy()
    .reshape(len(volcano_t2_img_data), 110, 110)
)

volcano_t3_img_data = (
    total_img
    .loc[(total_lbl['Volcano?'] == 1) &
        (total_lbl['Type'] == 3)]
)
volcano_t3_img_data = (
    volcano_t3_img_data
    .to_numpy()
    .reshape(len(volcano_t3_img_data), 110, 110)
)

volcano_t4_img_data = (
    total_img
    .loc[(total_lbl['Volcano?'] == 1) &
        (total_lbl['Type'] == 4)]
)
volcano_t4_img_data = (
    volcano_t4_img_data
    .to_numpy()
    .reshape(len(volcano_t4_img_data), 110, 110)
)

# ------------------------------------------------------------------------------
## DATA AUGMENTATION OF VOLCANO IMAGES

# Need to return dataframes that can be stored row-wise
volcano_t1_augmented_img = pd.DataFrame(
    augmentation(volcano_t1_img_data)
    .reshape(len(volcano_t1_img_data)*5, 110**2))
volcano_t1_augmented_lbl = pd.DataFrame(np.full(len(volcano_t1_img_data), 1))

volcano_t2_augmented_img = pd.DataFrame(
    augmentation(volcano_t2_img_data)
    .reshape(len(volcano_t2_img_data)*5, 110**2))
volcano_t2_augmented_lbl = pd.DataFrame(np.full(len(volcano_t2_img_data), 2))

volcano_t3_augmented_img = pd.DataFrame(
    augmentation(volcano_t3_img_data)
    .reshape(len(volcano_t3_img_data)*5, 110**2))
volcano_t3_augmented_lbl = pd.DataFrame(np.full(len(volcano_t3_img_data), 3))

volcano_t4_augmented_img = pd.DataFrame(
    augmentation(volcano_t4_img_data)
    .reshape(len(volcano_t4_img_data)*5, 110**2))
volcano_t4_augmented_lbl = pd.DataFrame(np.full(len(volcano_t4_img_data), 4))

# Create the compound data set
total_augmented_img = pd.concat(
    [volcano_t1_augmented_img, volcano_t2_augmented_img,
    volcano_t3_augmented_img, volcano_t4_augmented_img])

total_augmented_lbl = pd.concat(
    [volcano_t1_augmented_lbl, volcano_t2_augmented_lbl,
    volcano_t3_augmented_lbl, volcano_t4_augmented_lbl])
