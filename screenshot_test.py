from PIL import Image, ImageEnhance
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
import cv2

# Open image
img = cv2.imread('venus_Dome.png', cv2.IMREAD_GRAYSCALE)
diff_v = img.shape[1] - img.shape[0]
img = img[:,diff_v:]

print('Original Dimensions : ', img.shape)
# Resize image
scale_percent = (110/img.shape[0])*100.5 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape)


# Load model and test
model = tf.keras.models.load_model('models/CNN_model_v1_augmented_data.h5')
resized = (resized/255).reshape(1, 110, 110, 1).astype('float32')
pred = model.predict(resized)
