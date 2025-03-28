import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iteration = 100
sample = 2  # Fixed sample index
lim = 100

# Load the data
data_no_orca = np.load('../final_shapes/flow_no_orca/all_pcs.npy')
data_orca = np.load('../final_shapes/flow_orca/all_pcs.npy')
data_diff = np.load('../final_shapes/diffusion/all_pcs.npy')
data_final_orca = np.load('../final_shapes/flow_final_orca/all_pcs.npy')
data_final_orca = data_final_orca * 100 / 3

# Define the datasets to plot
datasets = [data_no_orca, data_orca, data_diff, data_final_orca]
titles = ['No ORCA', 'With ORCA', 'Diffusion', 'Final ORCA']

fig, axs = plt.subplots(1, 4, figsize=(32, 8), subplot_kw={'projection': '3d'})

for idx, (dataset, title) in enumerate(zip(datasets, titles)):
    ax = axs[idx]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    if(idx == 2):
        sample = sample +1
    x = dataset[sample, iteration, :, 0]
    y = dataset[sample, iteration, :, 1]
    z = dataset[sample, iteration, :, 2]

    ax.scatter(x, y, z)
    ax.set_title(title, fontsize=15)
    ax.grid(False)
    ax.axis('off')

plt.tight_layout()
plt.show()
