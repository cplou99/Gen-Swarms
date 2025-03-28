import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to calculate pairwise distances between points
def calculate_distances(points):
    num_points = points.shape[0]
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(points[i] - points[j])
            distances[i, j] = distance
            distances[j, i] = distance
    return distances

# Parameters
iteration = 100
sample = 2  # Fixed sample index
lim = 100
threshold = 0.03 * 2 *1.1* 100 / 3

# Load the data
data_no_orca = np.load('../final_shapes/flow_no_orca/all_pcs.npy')
data_orca = np.load('../final_shapes/flow_orca/all_pcs.npy')
data_diff = np.load('../final_shapes/diffusion/all_pcs.npy')
data_final_orca = np.load('../final_shapes/flow_final_orca/all_pcs.npy')

ref_flow = np.load('../final_shapes/flow_no_orca/refs.npy')
ref_diff = np.load('../final_shapes/diffusion/refs.npy')

# Scale the data
data_final_orca = data_final_orca * 100 / 3

# Convert lists to NumPy arrays if they are not already
goals = np.array([ref_flow, ref_flow, ref_diff, ref_flow])


datasets = [data_orca, data_no_orca, data_diff, data_final_orca]
titles = ['Gen-Swarms', 'Matching Flow', 'Diffusion', 'MF + ORCA']

# Create 3D subplots
fig, axs = plt.subplots(1, 4, figsize=(32, 8), subplot_kw={'projection': '3d'})

for idx, (dataset, title) in enumerate(zip(datasets, titles)):
    ax = axs[idx]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    
    # Adjust sample index for the third dataset
    if idx == 2:
        sample += 1
    
    
    x = dataset[sample, iteration, :, 0]
    y = dataset[sample, iteration, :, 1]
    z = dataset[sample, iteration, :, 2]
    
    x_goal = goals[sample, :, 0]
    y_goal = goals[sample, :, 1]
    z_goal = goals[sample, :, 2]
    
    distances = calculate_distances(dataset[sample, iteration, :, :])
    collision_matrix = distances < threshold
    np.fill_diagonal(collision_matrix, False)  # Ignore self-collisions
    unique_collisions = np.any(collision_matrix, axis=1)
    
    # Colors for plotting
    colors = np.array([[0, 1, 0] for _ in range(len(x))])  # Default color: green
    colors[unique_collisions] = [1, 0, 0]  # Collision color: red
    
    # Scatter plot for dataset points
    ax.scatter(x, y, z, c=colors, s=5, marker='o')
    
    # Scatter plot for goal points
    #ax.scatter(x_goal, y_goal, z_goal, c='b', s=2, marker='o')
    
    ax.set_title(title, fontsize=15)
    ax.grid(False)
    ax.axis('off')

plt.tight_layout()
plt.show()
