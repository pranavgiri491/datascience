import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv(r"D:\ids2 project\Book1.csv")

fig=plt.figure()
#syntax for 3 D projection
ax =plt.axes(projection='3d')


z = np.linspace(0, 1, 100)
x = z * np.sin(24 * z)
y = z * np.cos(24 * z)

# defining the x,y,z axis
ax.set_xlabel('X-axis qty')
ax.set_ylabel('Y-axis revenue')
ax.set_zlabel('Z-axis marketprice')

#plot 
ax.plot3D(x, y, z, 'green')
ax.set_title('3D line plotting')
plt.show()
