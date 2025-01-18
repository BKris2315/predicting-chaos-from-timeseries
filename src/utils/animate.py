import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D 

# Font properties for the title
family = 'Myriad Pro'
title_font = fm.FontProperties(style='normal', size=20, weight='normal', stretch='normal')

save_folder = '../../figs/lorenz-animate'
os.makedirs(save_folder, exist_ok=True)

# Define the initial system state (aka x, y, z positions in space)
initial_state = [0.1, 0, 0]

# Define the system parameters sigma, rho, and beta
sigma = 10.
rho   = 28.
beta  = 8./3.

# Define the time points to solve for, evenly spaced between the start and end times
start_time = 1
end_time = 60
interval = 100
time_points = np.linspace(start_time, end_time, end_time * interval)

# Define the Lorenz system
def lorenz_system(current_state, t):
    x, y, z = current_state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Plot the system in 3 dimensions
def plot_lorenz(xyz, n):
    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1,1,1,1))
    ax.yaxis.set_pane_color((1,1,1,1))
    ax.zaxis.set_pane_color((1,1,1,1))
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    ax.plot(x, y, z, color='g', alpha=0.7, linewidth=0.7)
    ax.set_xlim((-30,30))
    ax.set_ylim((-30,30))
    ax.set_zlim((0,50))
    ax.set_title('Lorenz system attractor', fontproperties=title_font)
    
    plt.savefig(f"{save_folder}/{n:03d}.png", dpi=60, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# Return a list in iteratively larger chunks
def get_chunks(full_list, size):
    size = max(1, size)
    chunks = [full_list[0:i] for i in range(1, len(full_list) + 1, size)]
    return chunks

# Get incrementally larger chunks of the time points, to reveal the attractor one frame at a time
chunks = get_chunks(time_points, size=20)

# Get the points to plot, one chunk of time steps at a time, by integrating the system of equations
points = [odeint(lorenz_system, initial_state, chunk) for chunk in chunks]

# Plot each set of points, one at a time, saving each plot
for n, point in enumerate(points):
    plot_lorenz(point, n)

# Create an animated GIF of all the plots
images = [Image.open(image) for image in glob.glob(f"{save_folder}/*.png")]
gif_filepath = f"{save_folder}/animated-lorenz-attractor.gif"

# Save as an animated GIF
images[0].save(
    fp=gif_filepath,
    format='GIF',
    save_all=True,
    append_images=images[1:],
    duration=[100] + [5] * (len(images) - 2) + [100],  # Specify duration for each frame
    loop=0  # Infinite loop
)

print(f"GIF saved at {gif_filepath}")
