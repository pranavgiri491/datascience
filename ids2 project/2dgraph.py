import pandas as pd
import matplotlib.pyplot as plt

# Read the csv file
df = pd.read_csv(r"D:\ids2 project\Book1.csv")

# Plot a bar graph for product and qty
plt.figure(figsize=(10,6))
plt.bar(df['marketprice'], df['qty'])
plt.xlabel('Market Price')
plt.ylabel('Quantity')
plt.title('Market Price vs Quantity')
plt.show()