import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your dataset into a pandas DataFrame
df = pd.read_csv(r"D:\ids2 project\Book1.csv")

# Extract the x and y columns from your dataset
x = df['qty'].values
y = df['revenue'].values

# Define the function that you want to plot
def function(x, y, z):
    return np.cos(np.sqrt(x**2 + y**2 + z**2))

# Create a meshgrid of x and y values
X, Y = np.meshgrid(x, y)

# Generate z values
Z = function(X, Y, 0)  # Assuming z = 0 for simplicity, you can modify this as needed

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)

# Customize the viewing angle
ax.view_init(elev=30, azim=45)  # Adjust the elevation and azimuth angles for better viewing

# Add grid lines
ax.grid(True)

# Set the plot title and labels
ax.set_title('3D Contour Plot of function(x, y, z)')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('z', fontsize=12)

# Add a color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

# Show the plot
plt.show()
