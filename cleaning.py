import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
train_images = pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
        ("Project/data/volcano_data/train_images.csv"), header = None)
train_labels = pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
        ("Project/data/volcano_data/train_labels.csv"))
test_images = pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
        ("Project/data/volcano_data/test_images.csv"), header = None)
test_labels = pd.read_csv(("/Users/pab.nb/Desktop/Herts Master's ") +
        ("Project/data/volcano_data/test_labels.csv"))

# Print a few of the images
fig, ax = plt.subplots(2, 4, figsize=[5, 15], tight_layout=True)
indexes = np.random.randint(0, 7000, size=8)
m = 0
for i in range(4):
    for j in  range(2):
        ax[j, i].imshow(train_images.iloc[m].to_numpy().reshape(110,110))
        m += 1
plt.show()

#___________________________ NEED TO ANALYZE THE DATA _________________________#

# Study the balancing of the data set
total_img = pd.concat([train_images, test_images])
total_lbl = pd.concat([train_labels, test_labels])

# Veeery unbalanced
fig, ax = plt.subplots(1, 1, figsize=[10,10])
plt.title("Class distribution Magellanic Volcanoes")
cmap = plt.get_cmap("viridis")
ax.bar(np.unique(total_lbl["Volcano?"]),
            height=[len(total_lbl["Volcano?"].loc[total_lbl["Volcano?"] == 0]),
                    len(total_lbl["Volcano?"].loc[total_lbl["Volcano?"] == 1])],
            color=["orange", "green"], alpha=0.65,
            tick_label=["No Volcano", "Volcano"])
plt.show()

# From those that are 1 -> get the secondary key 
fig, ax = plt.subplots(1, 1, figsize=[10,10])
plt.title("Subclass distribution of Magellanic Volcanoes")
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
plt.show()
