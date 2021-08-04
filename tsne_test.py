from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loading_functions import load_cleaned_volcano_data

image_data_cleaned, label_data_cleaned = load_cleaned_volcano_data()
print('Data loaded.')

embedding = TSNE(n_components=2, init='pca', verbose=3).fit_transform(image_data_cleaned)
np.savetxt("tsne.csv", embedding, delimiter=",")

fig, ax = plt.subplots(1, 1, figsize=[20, 20])
plt.scatter(embedding[:, 0], embedding[:, 1], cmap=label_data_cleaned['Volcano?'])
plt.savefig('tsne.jpg', dpi=300)
