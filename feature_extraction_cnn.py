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

# Import data without the corrupted images
img_data, lbl_data = load_cleaned_volcano_data()
img_aug, lbl_aug = load_augmented_data()

# Do some cuts
img_non_volcano = img_data[lbl_data['Volcano?'] == 0]
lbl_non_volcano = lbl_data[lbl_data['Volcano?'] == 0]
img_data = pd.concat([img_non_volcano, img_aug], ignore_index=True)
img_lbl = pd.concat([lbl_non_volcano, lbl_aug], ignore_index=True)
print('Augmented data loaded')


# Normalize and convert to numpy
img_data = normalize_image_data(img_data)
print('Data normalized')

# Import the previous model that was obtained from the training with the
# regular images
model = tf.keras.models.load_model('CNN_model_v2_augmented_data.h5')

# -> Retrieve the convolutional layers from the previous model
for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue
    print(i , layer.name , layer.output.shape)

model_viz = Model(inputs=model.inputs , outputs=model.layers[1].output)

# Select normalized image
image = img_data[7945]
image = np.expand_dims(image, axis=0) # Need to expand dimensions so that the
# input looks like something from within the network

# Image of which the feature map will be computed
fig, ax = plt.subplots(1, 1, figsize=(15,15), tight_layout=False)
plt.suptitle("Type 1 Volcano - Example for the Feature Map", fontsize=30)
ax.imshow(image.reshape(110, 110), cmap='gray')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.savefig("figures/image_of_featuremap_augmented_v2.jpg", dpi=300)

# Compute and plot the feature map for one of the images
# from the second convo layer
features = model_viz.predict(image).reshape(54, 54, 32)
fig, ax = plt.subplots(6, 4, figsize=(20,15), tight_layout=False)
plt.suptitle("Feature map for a type 1 volcano", fontsize=30)
z = 0
for j in range(4):
    for i in range(6):
        ax[i, j].imshow(features[:,:,z],
         cmap='gray') #, vmin=min(features[:,:,i]), vmax=max(features[:,:,i]))
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])
        z += 1
plt.savefig("figures/feature_map_augmented_v2.jpg", dpi=300)
plt.close()
