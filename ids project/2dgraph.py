import pandas as pd
import numpy as np 
df = pd.read_csv("D:\\ids project\\Book1.csv")

import matplotlib.pyplot as plt

# Bar plot
df['Employee_name'].value_counts().plot(kind='bar')

plt.show()

# Histogram for Salary
df['Salary'].plot(kind='hist')

plt.show()