# REQUIRED
# Import necessary packages here.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# REQUIRED
# Helper function used to load datasets.
def load_dataset(filename):
  return pd.read_csv(filename)

# Change the string to match your course number and name.
output_str = "4361PedrozaCristian"

# REQUIRED
# Getting started with KNN.
#     1. Load data
#     2. Split data in training and test sets using SKLearn
#     3. Normalize/Standardize features as needed.
X = load_dataset('movie_data.csv')
Y = load_dataset('movie_labels.csv')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=17)

# REQUIRED
# This function should normalize a feature (column) of a dataset.
def normalize_feature(data, feature):
    normalized =  (data[feature] - np.min(data[feature]) / np.max(data[feature]) - np.min(data[feature].min()))
    return normalized

# This function should standardize a feature (column) of a dataset.
def standardize_feature(data, feature):
    standardized = (data[feature] - np.mean(data[feature]) / np.std(data[feature]))
    return standardized

# REQUIRED
# Define the kNN algorithm.

# This function should calculated the Euclidian distance between two datapoints.
def euclidian_distance(dp1, dp2):
  return np.linalg.norm(dp1 - dp2)

# This function should get the k nearest neighbors for a new datapoint.
def get_neighbors(x_train, new_dp, k):
    distances = [(index, euclidian_distance(new_dp, dp)) for index, dp in enumerate(x_train)]
    distances.sort(key=lambda x: x[1])
    neighbors = [x_train[index] for index, _ in distances[:k]]
    return neighbors
  
  
# This function should determine the class label for the current datapoint
# based on the majority of class labels of its k neighbors.
def predict_dp(neighbors, y_train):
    unique_labels, counts = np.unique(y_train, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    max_label = max(label_counts, key=label_counts.get)
    return max_label

# Use the kNN algorithm to predict the class labels of the test set
# with k = 3
k = 3
predictions = []
for datapoint in x_test:
    neighbors = get_neighbors(x_train, datapoint, k)
    predicted_label = predict_dp(neighbors, y_train)
    predictions.append(predicted_label)

# Calculate and print out the accuracy of your predictions!
correct = sum([y_true == y_pred for y_true, y_pred in zip(y_test, predictions)])
accuracy = (correct / len(y_test)) * 100
print(f"Accuracy: {accuracy:.2f}%")

