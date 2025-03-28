import plotly.graph_objs as go
import numpy as np

# Load your trajectory data
trajectory = np.load('../final_shapes/flow_no_orca/all_pcs.npy')[2]

# Define the number of frames and the number of points
num_frames = trajectory.shape[0]
num_points = trajectory.shape[1]

# Create lists to hold frames for animation
frames = []

# Iterate through each time step to create frames
for t in range(num_frames):
    points = trajectory[t]

    # Extract x, y, z coordinates for this frame
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a Scatter3d trace for this frame
    frame = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=10,
            color=z,  # Color by the z values
            colorscale='Magma',
            opacity=0.8
        )
    )

    # Add the trace to the list of frames
    frames.append(go.Frame(data=[frame], name=f'Frame {t}'))

# Create the initial frame (first time step)
initial_points = trajectory[0]
x = initial_points[:, 0]
y = initial_points[:, 1]
z = initial_points[:, 2]

# Create the base scatter plot
scatter = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=10,
        color=z,
        colorscale='Magma',
        opacity=0.8
    )
)

# Create layout with fixed axis ranges
layout = go.Layout(
    title='Trajectory Evolution',
    scene=dict(
        xaxis=dict(
            title='X-axis',
            range=[-100, 100],  # Set fixed range for x-axis
            autorange=False    # Ensure the axis does not auto-adjust
        ),
        yaxis=dict(
            title='Y-axis',
            range=[-100, 100],  # Set fixed range for y-axis
            autorange=False    # Ensure the axis does not auto-adjust
        ),
        zaxis=dict(
            title='Z-axis',
            range=[-100, 100],  # Set fixed range for z-axis
            autorange=False    # Ensure the axis does not auto-adjust
        )
    ),
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
                                  'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': True,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
)

# Create a figure with the initial data, frames, and layout
fig = go.Figure(data=[scatter], frames=frames, layout=layout)

# Show the plot
fig.show()
