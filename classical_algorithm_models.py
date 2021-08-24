import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_loading_functions import load_cleaned_volcano_data, load_augmented_data
from CNN_functions import normalize_image_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                                                classification_report)
from model_functions import get_weight
from sklearn.svm import LinearSVC
from sklearn.model_selection import ParameterGrid, GridSearchCV
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################################################################
###################### Data loading pipeline ###################################
################################################################################
# Import data wuthout the corrupted images
img_data, lbl_data = load_cleaned_volcano_data()
img_aug, lbl_aug = load_augmented_data()

# Do some cuts
img_data = pd.concat([img_data, img_aug], ignore_index=True)
img_lbl = pd.concat([lbl_data, lbl_aug], ignore_index=True)
print('Augmented data loaded')

# Do the train-test split and convert the labels to hot encoded vectores
X_train, X_test, y_train, y_test = train_test_split(
    img_data,
    img_lbl["Volcano?"], # Classify on whether there is a volcano or not
    test_size=0.33,
    random_state=1
)

# Also: normalize the data
X_train = X_train.to_numpy()/255.
X_test = X_test.to_numpy()/255.
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
print('Train/test split done')

################################################################################
########################## PARAMETER SPACE EXPLORATION #########################
################################################################################

### Linear support vector classifier ##
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
pd.DataFrame(lsvc_opt.cv_results_).to_csv('linear_svc_gridexploration_cv.csv',
                                                                    index=False)

### Random forest classifier ####
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
                     verbose=2, cv=2)
        .fit(X_train, y_train)
    )
pd.DataFrame(rfc_opt.cv_results_).to_csv('rfc_gridexploration_cv.csv',
                                                                    index=False)

################################################################################
######################## CREATE THE FINAL MODELS - TEST ########################
################################################################################

rfc_opt = pd.read_csv("../data/rfc_gridexploration_cv.csv")
lsvc_opt = pd.read_csv("../data/linear_svc_gridexploration_cv.csv")

optimized_rfc = rfc_opt[rfc_opt.rank_test_score == 1]
rfc_opt_dict = {
    'criterion': optimized_rfc.param_criterion.values[0],
    'n_estimators': optimized_rfc.param_n_estimators.values[0],
    'max_depth': optimized_rfc.param_max_depth.values[0],
    'min_samples_split': optimized_rfc.param_min_samples_split.values[0],
    'min_samples_leaf': optimized_rfc.param_min_samples_leaf.values[0],
    'class_weight': optimized_rfc.param_class_weight.values[0]
}
## Create latex table for thesis
pd.DataFrame(rfc_opt_dict).to_latex()
model_rfc = RandomForestClassifier(**rfc_opt_dict).fit(X_train, y_train)

optimized_lsvc = lsvc_opt[lsvc_opt.rank_test_score == 1]
lsvc_opt_dict = {
lsvc_param_dict = {
    'C': optimized_lsvc.param_C.values[0],
    'class_weight': optimized_lsvc.param_class_weight.values[0]
}
model_lsvc = LinearSVC(dual=False, **lsvc_param_dict).fit(X_train, y_train)
