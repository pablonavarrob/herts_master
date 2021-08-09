import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_loading_functions import load_cleaned_volcano_data
from CNN_functions import normalize_image_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                                                classification_report)
from model_functions import get_weight
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
print('Train/test split done')

### Class balancing
# # Adjust weight for all classes automatically inversely proportional to class
# # frequency
# weights = get_weight(lbl_data)
# balanced_weights = {
#     0: weights[0],
#     1: weights[1]
# }

### Try with SVC (Support Vector Classification)
# Do a parameter space exploration
svc_param_dict = {
    'kernel': ['rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': [100, 10, 10, 0.1, 0.01],
    'class_weight': [None, 'balanced']
}

# Create the parameter grid
param_grid_svc = ParameterGrid(svc_param_dict)

# Run the function for all the created dicts - store the results
for params in param_grid:
    # Train, get results from trainig, run test -> save all that info in a dict?
    params = param_grid[10]
    svc_classifier = SVC(verbose=1, n_jobs=-1, **params).fit(X_test, y_test)

### Try with random forest classifier
# Do a parameters space exploration
rfc_param_dit = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [300, 500, 700, 1000],
    'max_depth': [20, 30, 40, 50],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 3],
    'class_weight': [None, 'balanced']
}
param_grid_rfc = ParameterGrid(rfc_param_dit)

results = []
for i in range(len(param_grid_rfc)):
    clf = RandomForestClassifier(
        n_jobs=-1,
        verbose=1,
        **param_grid_rfc[i]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({
        'idx': i,
        'accuracy': acc
    })
