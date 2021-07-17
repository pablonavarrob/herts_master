import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from helper_functions import load_volcano_data, rotate_180, rotate_270

# Load the data
train_images, train_labels, test_images, test_labels = load_volcano_data()

# Remix the data for the augmentation
total_img = pd.concat([train_images, test_images])
total_lbl = pd.concat([train_labels, test_labels])

# ------------------------------------------------------------------------------
## SELECT THE IMAGES CONTAINING VOLCANOES FOR THE AUGMENTATION PROCEDURE ##

volcano_t1_img_data = (
    total_img
    .loc[
        (total_lbl['Volcano?'] == 1) &
        (total_lbl['Type'] == 1)
    ]
)

volcano_t2_img_data = (
    total_img
    .loc[
        (total_lbl['Volcano?'] == 1) &
        (total_lbl['Type'] == 2)
    ]
)

volcano_t3_img_data = (
    total_img
    .loc[
        (total_lbl['Volcano?'] == 1) &
        (total_lbl['Type'] == 3)
    ]
)

volcano_t4_img_data = (
    total_img
    .loc[
        (total_lbl['Volcano?'] == 1) &
        (total_lbl['Type'] == 4)
    ]
)

# ------------------------------------------------------------------------------
