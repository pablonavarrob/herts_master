# Test the trained model
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from data_loading_functions import load_cleaned_volcano_data, load_augmented_data
from CNN_functions import normalize_image_data, build_model_CNN_v3
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

# Build the model and compile it
# input_shape = (110, 110, 1)
model = build_model_CNN_v3((110, 110, 1))
# Define adam and compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Binary classification
    metrics=['acc', 'AUC']
)
model.summary()

# Fit the model to the data and get the results
history = model.fit(
    X_train, y_train,
    batch_size=100, epochs=45,
    verbose=True, validation_split=0.125, callbacks=None)

# Save model and weights to HDF5
model.save("models/CNN_model_v3_augmented_data.h5")
print("Saved model to disk")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('hist_cnn_v3_aug.csv', index=False)
