import numpy as np
import pandas as pd

def load_volcano_data(separate_test_training = True):
    """ Loads the Magellanic Volcano data for the analysis.

    Use the kwarg separate_test_training if one wants to have it all together
    as a dataframe and not separated into test and training sets. """

    train_images = pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
            ("Project/data/volcano_data/train_images.csv"), header = None)
    train_labels = pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
            ("Project/data/volcano_data/train_labels.csv"))
    test_images = pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
            ("Project/data/volcano_data/test_images.csv"), header = None)
    test_labels = pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
            ("Project/data/volcano_data/test_labels.csv"))


    if separate_test_training == True:
        return train_images, train_labels, test_images, test_labels

    else:
        return pd.concat([train_images, test_images]), pd.concat([train_labels, test_labels])


def remove_corrupted_images(image_data, print_all_corrupt = False):

    """ Removig corruped images is basedo on the fact that they have zones
    full of black pixels (clipped due to missing information, most probably
    due to transmission errors with the spacecraft). """

    corrupted_indexes = [idx for idx in range(len(image_data))
                                if 0 in np.unique(image_data.iloc[idx])]

    print("Found {} corruped images out of {}".format(len(corrupted_indexes),
                                                            len(image_data)))

    if print_all_corrupt == True:
        for index in corrupted_indexes:
            fig, ax = plt.subplots(1, 1, figsize=[5,5])
            ax.imshow(train_images.iloc[index].to_numpy().reshape(110, 110))
            plt.savefig("test/{}".format(index))
            plt.close()

    return image_data.drop(corrupted_indexes), corrupted_indexes


def load_volcano_data_cleaned(separate_test_training = True):
    """ Loads the Magellanic Volcano data for the analysis with the
    corrupted images removed.

    Use the kwarg separate_test_training if one wants to have it all together
    as a dataframe and not separated into test and training sets. """

    train_images_non_corrupted, idx_corrupted_train = remove_corrupted_images(
            pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
            ("Project/data/volcano_data/train_images.csv"), header = None)
        )
    train_labels_non_corrupted = pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
            ("Project/data/volcano_data/train_labels.csv")).drop(idx_corrupted_train)

    test_images_non_corrupted, idx_corrupted_test = remove_corrupted_images(
            pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
            ("Project/data/volcano_data/test_images.csv"), header = None)
        )
    test_labels_non_corrupted = pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
            ("Project/data/volcano_data/test_labels.csv")).drop(idx_corrupted_test)


    if separate_test_training == True:
        return train_images, train_labels, test_images, test_labels

    else:
        return pd.concat([train_images, test_images]), pd.concat([train_labels, test_labels])


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
