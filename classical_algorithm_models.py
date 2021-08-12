import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_loading_functions import load_cleaned_volcano_data
from CNN_functions import normalize_image_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                                                classification_report)
from model_functions import get_weight
from sklearn.svm import LinearSVC
from sklearn.model_selection import ParameterGrid, GridSearchCV
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Same data loading pipeline ##################################################
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

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
print('Train/test split done')

### Random forest classifier ###################################################
lsvc_param_dict = {
    'C': [[0.001], [0.01], [0.1], [1], [10]],
    'class_weight': [[None], ['balanced']],
    'dual': [[False]]
}
param_grid_lsvc = ParameterGrid(lsvc_param_dict)
lsvc = LinearSVC()
lsvc_opt = (
        GridSearchCV(lsvc, param_grid_lsvc,
                     scoring='roc_auc', n_jobs=-1,
                     verbose=2, cv=10)
        .fit(X_train, y_train)
    )
pd.DataFrame(lsvc_opt.cv_results_).to_csv('linear_svc_gridexploration_cv.csv', index=False)

### Random forest classifier ###################################################
# Do a parameters space exploration
rfc_param_dit = {
    'criterion': [['gini'], ['entropy']],
    'n_estimators': [[300], [500], [700], [1000]],
    'max_depth': [[20], [30], [40], [50]],
    'min_samples_split': [[2], [5]],
    'min_samples_leaf': [[1], [2], [3]],
    'class_weight': [[None], ['balanced']]
}
param_grid_rfc = ParameterGrid(rfc_param_dit)
rfc = RandomForestClassifier()
rfc_opt = (
        GridSearchCV(rfc, param_grid_rfc,
                     scoring='roc_auc', n_jobs=-1,
                     verbose=2, cv=3)
        .fit(X_train, y_train)
    )
pd.DataFrame(rfc_opt.cv_results_).to_csv('rfc_gridexploration_cv.csv', index=False)
