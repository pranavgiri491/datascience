import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\ids2 project\Book1.csv")

df.plot(kind='line', x='revenue', y='qty')

plt.title('Sales Over Time')
plt.xlabel('revenue')
plt.ylabel('qty')
plt.show()