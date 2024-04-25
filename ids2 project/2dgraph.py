import pandas as pd
import numpy as np 
df = pd.read_csv(r"D:\ids2 project\Book1.csv")

import matplotlib.pyplot as plt

# Bar plot
df['revenue'].value_counts().plot(kind='bar')

plt.show()

# Histogram for product 
df['product'].plot(kind='hist')

plt.show()