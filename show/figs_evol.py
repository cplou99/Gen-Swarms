import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.ticker import MaxNLocator
import time

%matplotlib qt
data = np.load('../results/flow_orca/all_pcs.npy')
sample = 20

data_size = data.shape
num_it = data_size[1] - 1
num_points = data_size[2]

start_it = 0
it_max = num_it

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_data = data[0, :, 0]
y_data = data[0, :, 1]
z_data = data[0, :, 2]


colors = [26/255,117/255,255/255] 
#colors = np.array([[255,210,77] for _ in range(len(x))])  # Default color diff
# Initialize empty plots for the animations
sc = ax.scatter(x_data, y_data, z_data, c=colors, s=5, marker='o')

# Set the axis labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Set the limits of the plot

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Set the limits of the plot
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
ax.set_zlim(-100,100)
ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
ax.zaxis.set_major_locator(MaxNLocator(nbins=3))

ax.grid(False)
ax.axis('off')
frames_to_stop = [1, 50, 75, 90, 99]
# Update function for the animation
def update(frame):
    # Update the data for each frame (e.g., move points)
    
        
    current_frame = start_it + frame
    x_data = data[sample,current_frame, :, 0]
    y_data = data[sample,current_frame, :, 1]
    z_data = data[sample,current_frame, :, 2]
    
    # Update the scatter plot
    sc._offsets3d = (x_data, y_data, z_data)
    ax.set_title("Iteration " + str(current_frame) + " Sample " + str(sample), fontsize=12)
    
    
    #pause
    if current_frame in frames_to_stop :
        time.sleep(1)  # Pause for 5 seconds
    return sc,

# Create the animation
animation = FuncAnimation(fig, update, frames=it_max, interval=50)

#animation.save('/home/pablo/Desktop/python/diffusion-master/last_results/diffusion_2.mp4', writer='ffmpeg', fps=5)

# Show the legend
# Show the plot
plt.show()
