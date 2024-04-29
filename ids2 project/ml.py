# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'D:\ids2 project\Book1.csv')

# Assuming 'X' contains the features for clustering
# Replace 'qty', 'revenue', 'marketprice' with actual column names
X = data[['qty', 'revenue', 'marketprice']]
# Handling missing values by imputing with the mean of each feature
X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the silhouette score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot silhouette scores to find optimal number of clusters
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()

# Choose the optimal number of clusters based on the plot or business requirements
n_clusters = 3  # Replace with the chosen number of clusters

# Train the k-means model with the chosen number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Assign cluster labels to the original data
data['cluster'] = kmeans.labels_

# Split data into features (X) and target variable (y)
X_train, X_test, y_train, y_test = train_test_split(data[['cluster']], data['revenue'], test_size=0.2, random_state=42)

# Train linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Optionally, you can also analyze the coefficients of the linear regression model
print("Linear Regression Coefficients:", lr.coef_)

# Optionally, you can visualize the actual vs. predicted values
# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual Revenue")
# plt.ylabel("Predicted Revenue")
# plt.title("Actual vs. Predicted Revenue")
# plt.show()

# Further analysis and interpretation of results can be done based on your specific business needs
