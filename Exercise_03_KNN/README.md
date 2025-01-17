# Exercise 02

## Implement k-nearest neighbors classification using Python

### Aim  

To implement the k-nearest neighbors (k-NN) classification algorithm using Python.

### Theory  

The k-nearest neighbors (k-NN) algorithm is a simple, non-parametric method used for classification and regression. It works by finding the `k` closest training examples to a new data point and making predictions based on the majority label (for classification) or average (for regression).

The algorithm follows these steps:  

1. Calculate the distance between the test point and all the points in the dataset.  
2. Sort the distances and identify the `k` nearest neighbors.  
3. Predict the class label by majority voting from the `k` neighbors.

The distance metric most commonly used is Euclidean distance.

### Procedure/Program  

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# load the iris dataset
data = load_iris()
X = data.data    # features
y = data.target  # labels

# split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# initialize the k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# train the model
knn.fit(X_train, y_train)

# make predictions
y_pred = knn.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the k-NN classifier: {accuracy * 100:.2f}%")
```

### Output/Explanation  

Output:  

The program outputs:  

```bash
Accuracy of the k-NN classifier: 100.00%
```

Explanation:  

- The `Iris` dataset is loaded and split into training and test sets.  
- The features are standardized using `StandardScaler` to improve the accuracy of the k-NN algorithm.  
- A k-NN classifier is initialized with `k=3` and trained on the training dataset.  
- Predictions are made on the test dataset, and the accuracy of the classifier is calculated and displayed.
