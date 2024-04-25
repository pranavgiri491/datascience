import numpy as np
import pandas as pd


# making data frame from csv file
df=pd.read_csv('D:\ids2 project\Book1.csv')
print(df)

# add some missing value
df.loc[2, 'revenue '] = np.nan
df.loc[3, ' qty'] = np.nan
df.loc[1, 'reps'] = np.nan

#print(df)




# Calculate mean, median, and maximum and minimum values for the revenue column
mean_revenue = df['revenue'].mean()
median_revenue = df['revenue'].median()
max_revenue = df['revenue'].max()
min_revenue = df['revenue'].min()

print("Mean revenue:", mean_revenue)
print("Median revenue:", median_revenue)
print("Maximum revenue:", max_revenue)
print("Minimum revenue:", min_revenue)


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = df.drop('revenue', axis=1)
y = df['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)