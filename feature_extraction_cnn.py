# For the feature extraction we define another model that takes an image
# as the input and outputs feature maps: intermediate representation of the
# layers after the first one.
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from data_loading_functions import load_cleaned_volcano_data, load_augmented_data
from CNN_functions import normalize_image_data, build_model_CNN_v1
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the previous model that was obtained from the training with the
# regular images
model = tf.keras.models.load_model('CNN_model_v1.h5')

# -> Retrieve the convolutional layers from the previous model
for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue
    print(i , layer.name , layer.output.shape)

model_viz = Model(inputs=model.inputs , outputs=model.layers[1].output)

# Select normalized image
image = img_data_normalized[0]
image = np.expand_dims(image, axis=0) # Need to expand dimensions so that the
# input looks like something from within the network

#calculating features_map
features = model_viz.predict(image).reshape(54, 54, 5)
fig, ax = plt.subplots(5, 1, figsize=(20,15))
for i in range(5):
    ax[i].imshow(features[:,:,i],
     cmap='gray') #, vmin=min(features[:,:,i]), vmax=max(features[:,:,i]))
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])
plt.savefig("feature_map.jpg", dpi=300)
plt.close()


plt.imshow(image.reshape(110, 110), cmap='gray')
plt.savefig("image_to_feature_map.jpg", dpi=300)

# It might be that we need to remove some columns/rows to return the
# correct shape for the convo filter -> use below code for that
# image = (image.reshape(110, 110))[1:109, 1:109].reshape(108, 108, 1)
