import numpy as np
import pandas as pd


# making data frame from csv file
df=pd.read_csv('book2.csv')
print(df)

# add some missing value

df.loc[2, 'Project_Id'] = np.nan
df.loc[3, 'Project_Id'] = np.nan
df.loc[4, 'Project_Id'] = np.nan

print(df)


# Calculate mean, median, and maximum and minimum values for the salary column

# Calculate the mean
mean_salary = df['Salary'].mean()
print(f'Mean Salary: {mean_salary}')

# Calculate the median
median_salary = df['Salary'].median()
print(f'Median Salary: {median_salary}')

# Calculate the maximum
max_salary = df['Salary'].max()
print(f'Maximum Salary: {max_salary}')

# Calculate the minimum
min_salary = df['Salary'].min()
print(f'Minimum Salary: {min_salary}')


# Calculate the mean, ignoring missing values
mean_salary = df['Salary'].dropna().mean()
print(f'Mean Salary: {mean_salary}')
#################################################################################
# Test and train the data

from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame and 'Salary' is the target column
X = df.drop('Salary', axis=1)
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame and 'Salary' is the target column
X = df.drop('Salary', axis=2)
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)
#initial the model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

#import minimaxscalar
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
minmax_scaler = MinMaxScaler()

# Fit and transform the features
X_scaled = minmax_scaler.fit_transform(X)



