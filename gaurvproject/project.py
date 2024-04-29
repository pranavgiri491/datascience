import pandas as pd
import numpy as np
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv('Airlines.csv')
print(df)

# Step 2: Define Features and Target Variable
df.drop(columns=['Delay'],inplace=True)
print("\nDataset After Dropping the 'Delay' Column:")
print(df)

#identify missing values
missing_values = df.isnull().sum()
print("\ncount of missing values")
print(missing_values)


# step 3: Calculate mean, maximum, and minimum values of each column
print("\nCalculate mean, maximum, and minimum values of each column")
id_mean = df['id'].mean()
id_max = df['id'].max()
id_min = df['id'].min()

# Display the results
print("Mean values:")
print("id mean:", id_mean)
print("\nMaximum values:")
print("id max:", id_max)
print("\nMinimum values:")
print("id min:", id_min)

print("\nmean, maximum, minimum values of Flight column")
Flight_mean = df['Flight'].mean()
Flight_max = df['Flight'].max()
Flight_min = df['Flight'].min()

# Display the results
print("Mean values:")
print("Flight mean:", Flight_mean)
print("\nMaximum values:")
print("Flight max:", Flight_max)
print("\nMinimum values:")
print("Flight min:", Flight_min)

print("\nmean, maximum, minimum values of Flight column")
DayOfWeek_mean = df['DayOfWeek'].mean()
DayOfWeek_max = df['DayOfWeek'].max()
DayOfWeek_min = df['DayOfWeek'].min()

# Display the results
print("Mean values:")
print("DayOfWeek mean:", DayOfWeek_mean)
print("\nMaximum values:")
print("DayOfWeek max:", DayOfWeek_max)
print("\nMinimum values:")
print("DayOfWeek min:", DayOfWeek_min)

print("\nmean, maximum, minimum values of Time column")
Time_mean = df['Time'].mean()
Time_max = df['Time'].max()
Time_min = df['Time'].min()

# Display the results
print("Mean values:")
print("Time mean:", Time_mean)
print("\nMaximum values:")
print("Time max:", Time_max)
print("\nMinimum values:")
print("Time min:", Time_min)

print("\nmean, maximum, minimum values of Length column")
Length_mean = df['Length'].mean()
Length_max = df['Length'].max()
Length_min = df['Length'].min()

# Display the results
print("Mean values:")
print("Length mean:", Length_mean)
print("\nMaximum values:")
print("Length max:", Length_max)
print("\nMinimum values:")
print("Length min:", Length_min)

# Handle missing values (replace with mean of each column)
df.fillna(value={'id': id_mean, 'Flight': Flight_mean, 'DayOfWeek': DayOfWeek_mean, 'Time': Time_mean, 'Length': Length_max +1}  , inplace=True)
print("\nDataset after handling missing values:")
print(df)


# Apply Label Encoding
label_encoder = LabelEncoder()
df['Time_LabelEncoded'] = label_encoder.fit_transform(df['Time'])
print("\nDataset After applying LabelEncoder")
print(df)


# Apply One Hot Encoding
onehotencoder = OneHotEncoder()
DayOfWeek_onehot = onehotencoder.fit_transform(df[['DayOfWeek']]).toarray()
DayOfWeek_onehot_df = pd.DataFrame(DayOfWeek_onehot, columns = onehotencoder.get_feature_names(['DayOfWeek']))

# Concatenate the original dataframe with the one hot encoded dataframe
df = pd.concat([df, DayOfWeek_onehot_df], axis=1)

# Drop the original 'DayOfWeek' column
df.drop(columns=['DayOfWeek'], inplace=True)

print("\nDataset After applying OneHotEncoder")
print(df)