from sklearn.metrics import (f1_score, confusion_matrix,
                             roc_curve)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_loading_functions import load_cleaned_volcano_data, load_augmented_data
from CNN_functions import normalize_image_data, build_model_CNN_v2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


################################################################################
###################### Data loading pipeline ###################################
################################################################################
# Import data wuthout the corrupted images
img_data, lbl_data = load_cleaned_volcano_data()
img_aug, lbl_aug = load_augmented_data()

# Do some cuts
img_data = pd.concat([img_data, img_aug], ignore_index=True)
img_lbl = pd.concat([lbl_data, lbl_aug], ignore_index=True)
print('Augmented data loaded')

# Normalize and convert to numpy
img_data_normalized = normalize_image_data(img_data)
print('Data normalized')

# Do the train-test split and convert the labels to hot encoded vectores
X_train, X_test, y_train, y_test = train_test_split(
    img_data_normalized,
    img_lbl["Volcano?"], # Classify on whether there is a volcano or not
    test_size=0.33,
    random_state=1
)

# Also: normalize the data
X_train = X_train/255.
X_test = X_test/255.
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
print('Train/test split done')

# Import the model trained on augmented data
model = tf.keras.models.load_model('models/CNN_model_v2_augmented_data.h5')

# Show the confusion matrix
class_names = ['No Volcano', 'Volcano']
y_pred = tf.greater(model.predict(X_test), 0.95).numpy()
# Here in can use threshold
# when usuing tf.greater(predition, float_threshold)
cm = confusion_matrix(y_test, y_pred.astype(int))

sns.heatmap(
    cm, annot=True, xticklabels=class_names, yticklabels=class_names,
    fmt='d', cmap='Blues'
)
plt.show()
