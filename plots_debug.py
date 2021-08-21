import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loading_functions import load_raw_volcano_data, load_augmented_data

train_images, train_labels, test_images, test_labels = load_raw_volcano_data()

### DATA AUGMENTATION PLOTS ####################################################
aug_img, aug_lbl = load_augmented_data()
aug_img = aug_img.to_numpy().reshape(2824, 110, 110)
fig, ax = plt.subplots(2, 3, figsize=[15,10], tight_layout=True)
j = 0
for i in range(3):
    ax[0, i].imshow(aug_img[j], cmap='gray')
    ax[0, i].set_xticklabels([])
    ax[0, i].set_yticklabels([])
    ax[1, i].imshow(aug_img[j + 1], cmap='gray')
    ax[1, i].set_xticklabels([])
    ax[1, i].set_yticklabels([])
    j += 2
ax[0,0].set_ylabel('Raw', rotation=90, fontsize=26)
ax[1,0].set_ylabel('Augmented', rotation=90, fontsize=26)
plt.savefig('figures/augmented_example.png', dpi=150)

### OVERVIEW PLOTS ############################################################
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
