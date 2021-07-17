import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import load_volcano_data, rotate_180, rotate_270

train_images, train_labels, test_images, test_labels = load_volcano_data()

### DATA AUGMENTATION PLOTS #####
a = train_images.iloc[0].to_numpy().reshape(110, 110)
fig, ax = plt.subplots(2, 2, figsize=[10, 10], tight_layout=True)
fig.suptitle("Image augmentation: rotation", fontsize=20)
ax[0, 0].imshow(a)
ax[0, 0].title.set_text("Original image")
ax[0, 1].imshow(np.rot90(a))
ax[0, 1].title.set_text("90 deg. rotation")
ax[1, 0].imshow(rotate_180(a))
ax[1, 0].title.set_text("180 deg. rotation")
ax[1, 1].imshow(rotate_270(a))
ax[1, 1].title.set_text("270 deg. rotation")
plt.show()

fig, ax = plt.subplots(2, 1, figsize=[10, 10], tight_layout=True)
fig.suptitle("Image augmentation: mirroring", fontsize=20)
ax[0].imshow(np.fliplr(a))
ax[0].title.set_text("Mirrored image: horizontal axis")
ax[1].imshow(np.flipud(a))
ax[1].title.set_text("Mirrored image: vertical axis")
plt.show()
