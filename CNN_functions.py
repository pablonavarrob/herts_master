import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
import pandas as pd
from tensorflow import keras

def normalize_image_data(image_data):
    """ Normalize the pixel values from being 0 to 255 to a range between
    0 and 100 and returns a reshaped array if it is inputted as a
    DataFrame """

    if isinstance(image_data, pd.DataFrame):
        # Convert to array:
        image_data = image_data.to_numpy().reshape(len(image_data), 110, 110)

    else:
        pass

    return (image_data/255).astype('float32')

def build_model_CNN_v1(INPUT_SHAPE):
    """ Constructs the object model for the convolutional neural network
    through sequential layers. """

    model = Sequential()

    # The main architecture for the preliminary version will be based on a
    # standard construct from the Deep Learning book: 3 convolutional layers
    # and a fully connected layer at the end

    # Block 1
    model.add(Conv2D(3, 3), input_shape=INPUT_SHAPE, activation='relu')
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # Block 2
    model.add(Conv2D(4, 4), activation='relu')
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.3))

    # Block 3
    model.add(Conv2D(6, 6), activation='relu')
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.35))

    # Block 4
    model.add(Conv2D(8, 8), activation='relu')
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.4))

    # Dense layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation='sigmoid')) # Classification layer

    return model
