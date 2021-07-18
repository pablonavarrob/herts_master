import pandas as pd
from helper_functions import load_volcano_data
from cleaning_functions import remove_corrupted_images

total_img, total_lbl = load_raw_volcano_data(separate_test_training=False)


image_data_non_corrupted, idx_corrupted_image = remove_corrupted_images(

train_labels_non_corrupted = total_lbl.drop(idx_corrupted_image)
