from sklearn.metrics import (confusion_matrix,
                             roc_curve, plot_confusion_matrix,
                             plot_roc_curve)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loading_functions import load_cleaned_volcano_data, load_augmented_data
from CNN_functions import normalize_image_data, build_model_CNN_v2
import tensorflow as tf
import os
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
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


# One good way to measure performance for imbalanced datasets is the
# F-score, which combines precision and recall ina single measure.
# F-score is the harmonic mean of both quantities.

# Two types of curves:
# - Precision-Recall score -> average precision
# - Receiver Operating Characteristic -> Area Under Curve

# Both of them come from the confusion matrix
# -> To better choose parameters for the imbalanced data set, change
# the decision threshold from defeault to something that better represents
# the data.

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
    'class_weight': 'balanced'
}
## Create latex table for thesis
model_rfc = RandomForestClassifier(**rfc_opt_dict).fit(X_train, y_train)
rfc_pred = model_rfc.predict(X_test)

optimized_lsvc = lsvc_opt[lsvc_opt.rank_test_score == 1]
lsvc_opt_dict = {
    'C': optimized_lsvc.param_C.values[0],
    'class_weight': None
}
model_lsvc = LinearSVC(dual=False, **lsvc_opt_dict).fit(X_train, y_train)
lsvc_pred = model_lsvc.predict(X_test)
# Show the confusion matrix
# plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(1, 2, figsize=[14, 8], tight_layout=True, sharey=True)
class_names = ['No Volcano', 'Volcano']
plot_confusion_matrix(model_lsvc, X_test, y_test,
    display_labels=class_names, cmap=plt.cm.Blues,
    normalize=None, ax=ax[0], colorbar=False)
plot_confusion_matrix(model_rfc, X_test, y_test,
    display_labels=class_names, cmap=plt.cm.Blues,
    normalize=None, ax=ax[1], colorbar=False)
ax[0].set_title('Linear Support Vector Classifier', fontsize=20)
ax[1].set_title('Random Forest Classifier', fontsize=20)
plt.savefig('figures/confusion_matrices_result.png', dpi=150)

# Show the ROC curve
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(1, figsize=[12, 8], tight_layout=True) # , sharey=True)
class_names = ['No Volcano', 'Volcano']
plot_roc_curve(model_lsvc, X_test, y_test, ax=ax)
plot_roc_curve(model_rfc, X_test, y_test, ax=ax)
# ax[0].set_title('Linear Support Vector Classifier', fontsize=20)
# ax[1].set_title('Random Forest Classifier', fontsize=20)
ax.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.legend()
plt.savefig('figures/rocauc_classicmodels.png', dpi=150)

# Show the ROC curve
fig, ax = plt.subplots(1, figsize=[12, 8], tight_layout=True) # , sharey=True)
class_names = ['No Volcano', 'Volcano']
plot_precision_recall_curve(model_lsvc, X_test, y_test, ax=ax)
plot_precision_recall_curve(model_rfc, X_test, y_test, ax=ax)
plt.legend()
plt.savefig('figures/apcurve_classicmodels.png', dpi=150)
