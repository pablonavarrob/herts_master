from sklearn.metrics import (f1_score, confusion_matrix,
                             roc_curve, plot_confusion_matrix,
                             plot_roc_curve)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Show the confusion matrix 
class_names = ['No Volcano', 'Volcano']
disp = plot_confusion_matrix(clf, X_test, y_test,
    display_labels=class_names, cmap=plt.cm.Blues, normalize=None)
disp.ax_.set_title('Confusion Matrix')
plt.show()

# Show the ROC curve
roc = plot_roc_curve(clf, X_test, y_test)
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.legend()
roc.ax_.set_title('Receiver-Operator Characteristic')
plt.show()
