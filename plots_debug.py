import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loading_functions import load_raw_volcano_data

train_images, train_labels, test_images, test_labels = load_raw_volcano_data()

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

### OVERVIEW PLOTS #####
# Study the balancing of the data set
total_img, total_lbl = load_raw_volcano_data(False)
plt.rcParams.update({'font.size': 18})

# Veeery unbalanced
fig, ax = plt.subplots(1, 1, figsize=[15,10])
# plt.title("Class distribution Magellanic Volcanoes", fontsize=24)
cmap = plt.get_cmap("viridis")
ax.bar(np.unique(total_lbl["Volcano?"]),
            height=[len(total_lbl["Volcano?"].loc[total_lbl["Volcano?"] == 0]),
                    len(total_lbl["Volcano?"].loc[total_lbl["Volcano?"] == 1])],
            color=["orange", "green"], alpha=0.65,
            tick_label=[r"No Volcano", "Volcano"])
plt.savefig("figures/class_distribution.jpg", dpi=300)

# From those that are 1 -> get the secondary key
fig, ax = plt.subplots(1, 1, figsize=[18,10])
# plt.title("Subclass distribution of Magellanic Volcanoes", fontsize=24)
cmap = plt.get_cmap("viridis")
ax.bar([1, 2, 3, 4],
        height=[len(total_lbl.loc[total_lbl["Type"] == 1]),
                len(total_lbl.loc[total_lbl["Type"] == 2]),
                len(total_lbl.loc[total_lbl["Type"] == 3]),
                len(total_lbl.loc[total_lbl["Type"] == 4])],
        color=["green", "orange", "yellow", "red"], alpha=0.65,
        tick_label=[r"$p \approx 0.98$",
                    r"$p \approx 0.80$",
                    r"$p \approx 0.60$",
                    r"$p \approx 0.50$"])
plt.savefig("figures/subclass_distribution.jpg", dpi=300)
