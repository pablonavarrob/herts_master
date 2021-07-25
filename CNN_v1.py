import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from data_loading_functions import load_cleaned_volcano_data
from CNN_functions import normalize_image_data, build_model_CNN_v1
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import data wuthout the corrupted images
img_data, lbl_data = load_cleaned_volcano_data()
print('Data loaded')


# Normalize and convert to numpy
img_data_normalized = normalize_image_data(img_data)
print('Data normalized')


# Do the train-test split and convert the labels to hot encoded vectores
X_train, X_test, y_train, y_test = train_test_split(
    img_data_normalized,
    lbl_data["Volcano?"], # Classify on whether there is a volcano or not
    test_size=0.33,
    random_state=1
)

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
print('Train/test split done')


# Build the model and compile it
# input_shape = (110, 110, 1)
model = build_model_CNN_v1((110, 110, 1))
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',  # Binary classification
    metrics=['acc']
)
model.summary()

# Fit the model to the data and get the results
history = model.fit(
    X_train, y_train,
    batch_size=300, epochs=50,
    verbose=True, validation_split=0.33, callbacks=None
)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# Print the accuracy of the test data-set
score = model.evaluate(X_test, y_test, verbose=True)

plt.plot(history.history['acc'], label="train acc")
plt.plot(history.history['val_acc'], label="val acc")
plt.legend()
plt.savefig('acctraining.jpg', dpi=300)
