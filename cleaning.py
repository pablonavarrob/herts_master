import pandas as pd
from data_loading_functions import load_raw_volcano_data
from cleaning_functions import remove_corrupted_images

total_img, total_lbl = load_raw_volcano_data(separate_test_training=False)

# Call function; remove corrupted, get cleaned ones for later augmentation
image_data_non_corrupted, idx_corrupted_image = remove_corrupted_images(
        total_img.reset_index()
    )

train_labels_non_corrupted = total_lbl.drop(idx_corrupted_image)
