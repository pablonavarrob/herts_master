import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier

## Same data loading pipeline
# Import data without the corrupted images
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
