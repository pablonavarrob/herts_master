import pandas as pd
import numpy as np 

def get_weight(label_data):
    ''' Get the inverse frequency to get the weights for the
    samples in the analysis. '''

    n_samples = len(label_data)
    n_classes = len(label_data['Volcano?'].unique())
    weights = []
    for occ in range(n_classes):
        n_occurences = len(label_data[label_data['Volcano?'] == occ])
        weight = (n_samples)/(n_classes * n_occurences)
        weights.append(weight)

    return weights
