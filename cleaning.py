import pandas as pd
from data_loading_functions import load_raw_volcano_data
from cleaning_functions import remove_corrupted_images

SAVE_DATA_TO_FILE = False

total_img, total_lbl = load_raw_volcano_data(separate_test_training=False)
total_img = total_img.reset_index(drop=True)
total_lbl = total_lbl.reset_index(drop=True)

# Call function; remove corrupted, get cleaned ones for later augmentation
image_data_non_corrupted, idx_corrupted_image = remove_corrupted_images(
        total_img)
train_labels_non_corrupted = total_lbl.drop(idx_corrupted_image)

if SAVE_DATA_TO_FILE:
    image_data_non_corrupted.to_csv('image_data_non_corrupted.csv', index=False)
    train_labels_non_corrupted.to_csv('train_labels_non_corrupted.csv', index=False)
