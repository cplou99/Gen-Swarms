import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.ticker import MaxNLocator


%matplotlib qt

def plot_all(data, lim, title=""):
    # VELOCITY
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Changed to horizontal layout

    for i in range(3):
        axs[i].set_ylim(-150, 150)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['left'].set_color('lightgrey')
        axs[i].spines['bottom'].set_color('lightgrey')        
        axs[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
        axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        axs[i].tick_params(axis='both', which='major', labelsize=20)
        
    axs[0].plot(data[:, :, 0], marker='o')    
    axs[0].set_title('X', fontsize=18)    
    
    # Plot y values
    axs[1].plot(data[:, :, 1], marker='o')
    axs[1].set_title('Y', fontsize=18)

    
    # Plot z values
    axs[2].plot(data[:, :, 2], marker='o')
    axs[2].set_title('Z', fontsize=18)
    
    #fig.suptitle(title, fontsize=16)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
# Load the data
data_no_orca = np.load('../results/flow_no_orca/all_pcs.npy')
data_orca = np.load('../results/flow_orca/all_pcs.npy')
data_diff = np.load('../results/diffusion/all_pcs.npy')
data_final_orca = np.load('../results/flow_final_orca/all_pcs.npy')
data_final_orca = data_final_orca * 100 / 3

# Define the datasets to plot
datasets = [data_orca]#, data_no_orca, data_diff, data_final_orca]
titles = ['Gen-Swarms', 'Matching Flow', 'Diffusion', 'MF + ORCA']

datasets = [data_orca, data_diff]
titles = ['Gen-Swarms', 'Diffusion']

fig, axs = plt.subplots(1, 4, figsize=(32, 8), subplot_kw={'projection': '3d'})

delta_t = 1

data_size = data_orca.shape
num_it = data_size[1]
num_points = data_size[0]

it_max = num_it

sample = 15
particle = 11

#for one particle
#3d Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate the color gradient based on the number of iterations
gradient = np.linspace(0, 1, num_it)
lim = 200  # Adjusted the limit to make the axes bigger

for idx, (dataset, title) in enumerate(zip(datasets, titles)):
    ax = axs[idx]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    if(idx == 2):
        sample = sample + 1

    # 3D POSITION
    for particle in range(num_points):
        sc = ax.scatter(dataset[sample, :, particle, 0], dataset[sample, :, particle, 1], dataset[sample, :, particle, 2], c=gradient, cmap='magma', marker='o', s=20)  # Adjusted marker size

    ax.set_ylim(-150, 150)
    
    ax.set_zlim(-150, 150)
    
    ax.set_xlim(-150, 150)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set_title(title, fontsize=15)
    #ax.grid(False)
    #ax.axis('off')

    # 2D plots of X, Y, Z coordinates
    #plot_all(dataset[sample], 200, title)

plt.show()
