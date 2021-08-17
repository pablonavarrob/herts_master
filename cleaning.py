import pandas as pd
from data_loading_functions import load_raw_volcano_data
from cleaning_functions import remove_corrupted_images
import matplotlib.pyplot as plt

SAVE_DATA_TO_FILE = False

total_img, total_lbl = load_raw_volcano_data(separate_test_training=False)
total_img = total_img.reset_index(drop=True)
total_lbl = total_lbl.reset_index(drop=True)

# Call function; remove corrupted, get cleaned ones for later augmentation
image_data_non_corrupted, idx_corrupted_image = remove_corrupted_images(
        total_img)
train_labels_non_corrupted = total_lbl.drop(idx_corrupted_image)

# Plot some of the corrupted images
corrupted = total_img.iloc[idx_corrupted_image]
corrupted = corrupted.to_numpy().reshape(378, 110, 110)
fig, ax = plt.subplots(2, 3, figsize=[15, 10], tight_layout=True)
for i in range(3):
    ax[0, i].imshow(corrupted[i + 235], cmap='gray')
    ax[0, i].set_xticklabels([])
    ax[0, i].set_yticklabels([])
    ax[1, i].imshow(corrupted[i + 200], cmap='gray')
    ax[1, i].set_xticklabels([])
    ax[1, i].set_yticklabels([])
plt.savefig('figures/corruptedvolcanoes.png', dpi=150)


if SAVE_DATA_TO_FILE:
    image_data_non_corrupted.to_csv('image_data_non_corrupted.csv', index=False)
    train_labels_non_corrupted.to_csv('train_labels_non_corrupted.csv', index=False)
