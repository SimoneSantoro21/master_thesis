import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Script to visualize bves directions on unit sphere.
This script computes reverse polarity vectors and plots them along the original ones.
A solid angle is defined, and the number of directions laying within it is computed for each bvec.
"""

bvecs_single_shell = np.loadtxt('/Volumes/SEAGATE BAS/DTI_data/Dbs_080/Unnamed - 464/Converted_Nii_Files_1/6_ax_dti_b__1000_multiband.bvec')
bvals_single_shell = np.loadtxt('/Volumes/SEAGATE BAS/DTI_data/Dbs_080/Unnamed - 464/Converted_Nii_Files_1/6_ax_dti_b__1000_multiband.bval')

# Extract components for single-shell and computing reverse polarity vectors
x_single, y_single, z_single = bvecs_single_shell
x_reverse, y_reverse, z_reverse = -x_single, -y_single, -z_single

# Plot sphere for reference
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
Xs = np.outer(np.cos(u), np.sin(v))
Ys = np.outer(np.sin(u), np.sin(v))
Zs = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(Xs, Ys, Zs, color='gray', alpha=0.1, edgecolor='none')

# Plot single-shell points in green
ax.scatter(x_single, y_single, z_single, color='g', marker='*', s=50, label='Single-Shell')

# Plot single-shell points with reverse polarity in red
ax.scatter(x_reverse, y_reverse, z_reverse, color='r', marker='*', s=50, label='Reverse Polarity')

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gradient Directions on Unit Sphere')
plt.grid(True)
plt.show()

# Define the maximum angular threshold (in degrees converted to radians)
theta_max = 30  # degrees
theta_max_rad = np.deg2rad(theta_max)

# Combine original and reverse polarity directions
x_combined = np.concatenate((x_single, x_reverse))
y_combined = np.concatenate((y_single, y_reverse))
z_combined = np.concatenate((z_single, z_reverse))

# Loop through each single-shell direction as the center of the solid angle
for i in range(len(x_single)):
    solid_angle_center = np.array([x_single[i], y_single[i], z_single[i]])
    
    # Compute dot products between this center and all directions (original + reverse)
    dot_products = x_combined * solid_angle_center[0] + \
                   y_combined * solid_angle_center[1] + \
                   z_combined * solid_angle_center[2]
    
    # Convert dot products to angles (in radians)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
    
    # Count the number of directions within the angular threshold
    count_inside = np.sum(angles <= theta_max_rad)
    
    print(f'Single-shell direction {i + 1}: {count_inside} directions inside the {theta_max}° solid angle')
