import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class simple_cnn():
    ''' Defines a simple convolutional neural network and its parameters,
    it is not going to be very reusable but... '''

# Import data, shuffle it, and split between train and test
X_train = pd.read_csv('train_images.csv', header=None)
y_train = pd.read_csv('train_labels.csv')['Volcano?']
X_test = pd.read_csv('test_images.csv', header=None)
y_test = pd.read_csv('test_labels.csv')['Volcano?']

# normalize and convert to type, cast
X_train, X_test = (
    X_train/255.0).astype('float32'), (X_test/255.0).astype('float32')

# reshape
X_train = X_train.to_numpy().reshape((len(X_train), 110, 110, 1))
X_test = X_test.to_numpy().reshape((len(X_test), 110, 110, 1))

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

EPOCHS = 5
BATCH_SIZE = 512
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.Adam()
VALIDATION_SPLIT = 0.3
IMG_ROWS, IMG_COLS = 28, 28  # input image dimensions
INPUT_SHAPE = (110, 110, 1)
NB_CLASSES = 2  # number of outputs = number of digits

def build(input_shape, classes):
    # Create a sequential modelsa
    model = models.Sequential()
    # Add first convo layer, CONV -> RELU -> POOLING
    model.add(layers.Conv2D(20, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Second convolutional stage with CONV -> RELU -> POOLING
    # Increasing amount of filters in deeper layers is a common technique in deep learning
    model.add(layers.Convolution2D(50, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Flatten is the function that converts the pooled feature map
    # to a single column that is passed to the fully connected layer.
    # Flatten output, relu and softmax classification
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))

    return model

# initialize optimizer and model
model = build(input_shape=INPUT_SHAPE, classes=1) # calls our own function!!!!
model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER, metrics=['accuracy'])
model.summary()

# fit model to data
history = model.fit(X_train, y_train,
    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, y_test, verbose=VERBOSE)
