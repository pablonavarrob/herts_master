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

### Test the brightness and test the image increase/decease

inx = [8, 14, 45]
fig, ax = plt.subplots(3, 3, figsize=[15, 10])
for i in range(3):
    img = img_data_normalized[inx[i]]
    contrast = tf.reshape(random_contrast(
        img, lower=0.25, upper=1.85), (110, 110)).numpy()
    brightness = tf.reshape(random_brightness(img, 0.25), (110, 110)).numpy()
    # img = tf.image.resize(img_data_normalized[inx[i]], (110, 110))
    ax[i, 0].imshow(contrast)
    ax[i, 1].imshow(brightness)
    ax[i, 2].imshow(img.reshape(110, 110))
    ax[i, 1].set_xticklabels([])
    ax[i, 1].set_yticklabels([])
    ax[i, 0].set_xticklabels([])
    ax[i, 0].set_yticklabels([])
plt.savefig('augmentation_exposure.jpg', dpi=300)
