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
        image_data = image_data.to_numpy()

    else:
        pass

    return (image_data/255).reshape(len(image_data), 110, 110, 1).astype('float32')

def build_model_CNN_v1(INPUT_SHAPE):
    """ Constructs the object model for the convolutional neural network
    through sequential layers.

    The main architecture for the preliminary version will be based on a
    standard construct from the Deep Learning book: 3 convolutional layers
    and a fully connected layer at the end

    """

    model = Sequential()

    # Block 1
    model.add(Conv2D(5, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.3))

    # Block 2
    model.add(Conv2D(10, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.35))

    # Block 3
    model.add(Conv2D(15, (6, 6), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.4))

    # Block 4
    model.add(Conv2D(20, (8, 8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.45))

    # Dense layer
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1, activation='sigmoid')) # Classification layer

    return model

def build_model_CNN_v2(INPUT_SHAPE):
    """ Constructs the object model for the convolutional neural network
    through sequential layers. Version 2. """

    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.3))

    # Block 2
    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.35))

    # Block 3
    model.add(Conv2D(128, (6, 6), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.4))

    # Block 4
    model.add(Conv2D(128, (8, 8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.45))

    # Dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1, activation='sigmoid')) # Classification layer

    return model

def build_model_CNN_v3(INPUT_SHAPE):
    """ Constructs the object model for the convolutional neural network
    through sequential layers. Version 3. Minor changes in convolutional

    Adds an L2 regularization in the dense layer at the end. """

    model = Sequential()

    # Block 1
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.3))

    # Block 2
    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.35))

    # Block 3
    model.add(Conv2D(64, (6, 6), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.4))

    # Block 4
    model.add(Conv2D(128, (8, 8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.45))

    # Dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1, activation='sigmoid')) # Classification layer

    return model

# The benefits of using a convolutional network rather than a fully connected
# network is that to, theoretically achieve similar levels of accuracy,
# the fully connected network would have to be insanely massive compared to the
# convolutional network that we just showed above.
