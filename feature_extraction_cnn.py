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
from matplotlib import gridspec
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import data wuthout the corrupted images and also the augmented data
img_data, lbl_data = load_cleaned_volcano_data()
img_aug, lbl_aug = load_augmented_data()

# Do some cuts
img_data = pd.concat([img_data, img_aug], ignore_index=True)
img_lbl = pd.concat([lbl_data, lbl_aug], ignore_index=True)
print('Augmented data loaded')

# Normalize and convert to numpy
img_data_normalized = normalize_image_data(img_data)
print('Data normalized')

# Import the previous model that was obtained from the training with the
# regular images
model = tf.keras.models.load_model('models/CNN_model_v2_augmented_data.h5')

# -> Retrieve the convolutional layers from the previous model
for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue
    print(i , layer.name , layer.output.shape)

model_viz = Model(inputs=model.inputs , outputs=model.layers[0].output)

# Select normalized image
image = img_data_normalized[8]
image = np.expand_dims(image, axis=0) # Need to expand dimensions so that the
# input looks like something from within the network

# Image of which the feature map will be computed ##############################
fig, ax = plt.subplots(1, 1, figsize=(15,15), tight_layout=True)
ax.imshow(image.reshape(110, 110), cmap='gray')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.savefig("figures/image_of_featuremap.jpg", dpi=300)

# Compute and plot the feature map for one of the images #######################
# from the second convo layer
features = model_viz.predict(image).reshape(model.layers[0].output_shape[1],
                                            model.layers[0].output_shape[2],
                                            model.layers[0].output_shape[3])

nrow = 6
ncol = 4
fig = plt.figure(figsize=(ncol+1, nrow+1))
gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.0, hspace=0.0,
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
         left=0.5/(ncol+1), right=1-0.5/(ncol+1))
z = 0
for i in range(nrow):
    for j in range(ncol):
        ax = plt.subplot(gs[i,j])
        ax.imshow(features[:,:,z], cmap='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        z += 1
plt.savefig("figures/feature_map_augmented_v2.jpg", dpi=300)

# Plot the convolutional filters corresponding to 1st layer ####################
weights = model.layers[0].get_weights()[0][:,:,0,:]
z = 0
for i in range(nrow):
    for j in range(ncol):
        ax = plt.subplot(gs[i,j])
        ax.imshow(weights[:,:,z], cmap='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        z += 1
plt.savefig("figures/convolutional_filters_augmented_v2.jpg", dpi=300)
