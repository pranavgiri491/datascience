import pandas as pd
import matplotlib.pyplot as plt

# Read the csv file
df = pd.read_csv(r"D:\gaurvproject\Airlines.csv")

# Plot a bar graph for market price and quantity
plt.figure(figsize=(10,6))
plt.bar(df['Flight'], df['Length'])
plt.xlabel('Flight')
plt.ylabel('Length')
plt.title('flight and length')
plt.show()
