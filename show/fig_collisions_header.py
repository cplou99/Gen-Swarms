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
sample = 20  # Flow
#sample = 15  # Diff
lim = 100
threshold = 0.03 * 2 * 1 * 100 / 3

# Load the data
data_orca = np.load('../final_shapes/flow_orca/all_pcs.npy')

# Scale the data (if needed for this specific dataset)
#data_orca = data_orca * 100 / 3

# Use data_orca for plotting
dataset = data_orca
title = 'Gen-Swarms'

# Create a 3D plot for the specific dataset
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)

x = dataset[sample, iteration, :, 0]
y = dataset[sample, iteration, :, 1]
z = dataset[sample, iteration, :, 2]

distances = calculate_distances(dataset[sample, iteration, :, :])
collision_matrix = distances < threshold
np.fill_diagonal(collision_matrix, False)  # Ignore self-collisions
unique_collisions = np.any(collision_matrix, axis=1)

# Colors for plotting
colors = np.array([[26,117,255] for _ in range(len(x))])  # Default color flow
#colors = np.array([[255,210,77] for _ in range(len(x))])  # Default color diff
colors[unique_collisions] = [255, 0, 0]  # Collision color: red

# Scatter plot for dataset points
ax.scatter(x, y, z, c=colors/255, s=5, marker='o')

# Scatter plot for goal points
#ax.scatter(x_goal, y_goal, z_goal, c='b', s=2, marker='o')

ax.set_title(title, fontsize=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(False)
ax.axis('off')

plt.show()
