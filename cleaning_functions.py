import numpy as np
import pandas as pd
import matplotlib.pyplot as plt s

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
