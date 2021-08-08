import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from data_loading_functions import load_cleaned_volcano_data
from CNN_functions import normalize_image_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                                                classification_report)
from sklearn.decomposition import PCA

## Same data loading pipeline
# Import data without the corrupted images
img_data, lbl_data = load_cleaned_volcano_data()
print('Data loaded')

# Do the train-test split and convert the labels to hot encoded vectores
X_train, X_test, y_train, y_test = train_test_split(
    img_data,
    lbl_data["Volcano?"], # Classify on whether there is a volcano or not
    test_size=0.33,
    random_state=1
)

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
print('Train/test split done')

### Try with random forest classifier
clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=50,
    random_state=0,
    n_jobs=-1
)
clf.fit(X_train, y_train.ravel())
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
