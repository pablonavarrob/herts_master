import numpy as np
import pandas as pd


def load_raw_volcano_data(separate_test_training = True):
    """ Loads the Magellanic Volcano data for the analysis.

    Use the arg separate_test_training if one wants to have it all together
    as a dataframe and not separated into test and training sets. """

    train_images = pd.read_csv(
        ("../data/volcano_data/train_images.csv"), header = None)
    train_labels = pd.read_csv(("../data/volcano_data/train_labels.csv"))
    test_images = pd.read_csv(
        ("../data/volcano_data/test_images.csv"), header = None)
    test_labels = pd.read_csv(("../data/volcano_data/test_labels.csv"))

    total_img = pd.concat([train_images, test_images], ignore_index=True)
    total_lbl = pd.concat([train_labels, test_labels], ignore_index=True)

    if separate_test_training == True:
        return train_images, train_labels, test_images, test_labels

    else:
        return total_img, total_lbl

def load_cleaned_volcano_data():
    """ Loads the cleaned Magellanic Data for the analysis and to produce the
    augmented data set. """

    img_data = pd.read_csv("../data/image_data_non_corrupted.csv")

    lbl_data = pd.read_csv("../data/train_labels_non_corrupted.csv")

    return img_data, lbl_data

def load_augmented_data():
    """ Gets all of the augmented data sets and loads them into a separate
    array, leaving the originals out. """

    img_data_aug = pd.DataFrame()
    img_labl_aug = pd.DataFrame()

    for i in range(1, 5):
        load_aug_data_img = (
            pd.read_csv("../data/volcano_type{}_augmented_img.csv".format(i))
        )

        load_aug_data_lbl = (
            pd.read_csv("../data/volcano_type{}_augmented_lbl.csv".format(i))
        )

        img_data_aug = pd.concat(
            [img_data_aug, load_aug_data_img], ignore_index=True
        )
        img_labl_aug = pd.concat(
            [img_labl_aug, load_aug_data_lbl], ignore_index=True
        )

    return img_data_aug, img_labl_aug
