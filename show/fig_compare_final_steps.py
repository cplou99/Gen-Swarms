import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iteration = 100
sample = 2  # Fixed sample index
lim = 100

# Load the data
data_no_orca = np.load('../results/flow_orca_5/all_pcs.npy')
data_orca = np.load('../results/flow_orca_10/all_pcs.npy')
data_diff = np.load('../results/flow_orca_25/all_pcs.npy')
data_final_orca = np.load('../results/flow_orca/all_pcs.npy')

# Define the datasets to plot
datasets = [data_no_orca, data_orca, data_diff, data_final_orca]
titles = ['5', '10', '25', '100']

fig, axs = plt.subplots(1, 4, figsize=(32, 8), subplot_kw={'projection': '3d'})

for idx, (dataset, title) in enumerate(zip(datasets, titles)):
    ax = axs[idx]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    x = dataset[sample, -1, :, 0]
    y = dataset[sample, -1, :, 1]
    z = dataset[sample, -1, :, 2]

    ax.scatter(x, y, z)
    ax.set_title(title, fontsize=15)
    ax.grid(False)
    ax.axis('off')

plt.tight_layout()
plt.show()
