import pandas as pd
import numpy as np
from augmentation_functions import *
from data_loading_functions import load_cleaned_volcano_data


image_data_cleaned, label_data_cleaned = load_cleaned_volcano_data()
no_volcano = image_data_cleaned.loc[label_data_cleaned["Volcano?"] == 0]

for i in range(1, 5):
    volcano_ = (
        image_data_cleaned
        .loc[label_data_cleaned["Type"] == i]
        .to_numpy()
        .reshape(len(image_data_cleaned.loc[label_data_cleaned["Type"] == i]), 110, 110)
    )

    volcano_augmented_img, volcano_augmented_lbl = augmentation(
            volcano_,
            label_data_cleaned.loc[label_data_cleaned["Type"] == i]
        )

    volcano_augmented_img = pd.DataFrame(
        volcano_augmented_img
        .reshape(len(volcano_)*5, 110**2)
    )

    volcano_augmented_img.to_csv(
    "../data/volcano_type{}_augmented_img.csv".format(i),
    index=False)
    volcano_augmented_lbl.to_csv(
    "../data/volcano_type{}_augmented_lbl.csv".format(i),
    index=False)
