from sklearn.metrics import (f1_score, confusion_matrix,
                             roc_curve)
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from data_loading_functions import load_cleaned_volcano_data, load_augmented_data
from CNN_functions import normalize_image_data, build_model_CNN_v2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Import data wuthout the corrupted images
img_data, lbl_data = load_cleaned_volcano_data()
img_aug, lbl_aug = load_augmented_data()

# Normalize the non-augmented and convert to numpy
img_data_normalized = normalize_image_data(img_data)
print('Data normalized')

# Get the training set and the labels and all that
X_train, X_test, y_train, y_test = train_test_split(
    img_data_normalized,
    lbl_data["Volcano?"], # Classify on whether there is a volcano or not
    test_size=0.33,
    random_state=1
)

# Import the model trained on augmented data
model_cnn = tf.keras.models.load_model('CNN_model_v2_augmented_data.h5')

# Show the confusion matrix
class_names = ['No Volcano', 'Volcano']
y_pred = tf.greater(model_cnn.predict(X_test), 0.5).numpy()
# Here in can use threshold
# when usuing tf.greater(predition, float_threshold)
cm = confusion_matrix(y_test, y_pred.astype(int))

sns.heatmap(
    cm, annot=True, xticklabels=class_names, yticklabels=class_names,
    fmt='d', cmap='Blues'
)
plt.show()
# One good way to measure performance for imbalanced datasets is the
# F-score, which combines precision and recall ina single measure.
# F-score is the harmonic mean of both quantities.

# Two types of curves:
# - Precision-Recall score -> average precision
# - Receiver Operating Characteristic -> Area Under Curve

# Both of them come from the confusion matrix
# -> To better choose parameters for the imbalanced data set, change
# the decision threshold from defeault to something that better represents
# the data.
