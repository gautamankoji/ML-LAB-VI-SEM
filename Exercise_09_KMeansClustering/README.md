# Exercise 09

## Predict a classification for a case where VAR1 = 0.906 and VAR2 = 0.606 using k-means clustering with 3 centroids based on the given data

| VAR1  | VAR2  | CLASS |
| ----- | ----- | ----- |
| 1.713 | 1.586 | 0     |
| 0.180 | 1.786 | 1     |
| 0.353 | 1.240 | 1     |
| 0.940 | 1.566 | 0     |
| 1.486 | 0.759 | 1     |
| 1.266 | 1.106 | 0     |
| 1.540 | 0.419 | 1     |
| 0.459 | 1.799 | 1     |
| 0.773 | 0.186 | 1     |

### Aim

To predict the classification for a case where $\text{VAR1} = 0.906$ and $\text{VAR2} = 0.606$ using k-means clustering with 3 centroids based on the given dataset.

### Theory (Brief)

K-means clustering is an unsupervised learning algorithm used to partition data into $k$ clusters. Each cluster is represented by its centroid, and the algorithm iteratively refines the centroids to minimize intra-cluster variance. The steps are:

1. Initialize $k$ centroids randomly.
2. Assign each data point to the nearest centroid.
3. Recalculate the centroids based on the assigned points.
4. Repeat steps 2 and 3 until convergence.

The objective is to minimize the total within-cluster variance, given by:

$$
J = \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2
$$

Where:

- $k$: Number of clusters
- $C_i$: Cluster $i$
- $\mu_i$: Centroid of cluster $i$

### Procedure/Program

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# dataset
data = np.array([
    [1.713, 1.586],
    [0.180, 1.786],
    [0.353, 1.240],
    [0.940, 1.566],
    [1.486, 0.759],
    [1.266, 1.106],
    [1.540, 0.419],
    [0.459, 1.799],
    [0.773, 0.186]
])

# target classes for visualization (not used in k-means)
classes = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1])

# data point to classify
new_point = np.array([[0.906, 0.606]])

# normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
new_point_scaled = scaler.transform(new_point)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)
predicted_cluster = kmeans.predict(new_point_scaled)[0]
print(f"Centroids of clusters:\n{kmeans.cluster_centers_}")
print(f"The new point {new_point} belongs to cluster: {predicted_cluster}")
```

### Output/Explanation

Output:

```bash
Centroids of clusters:
[[-1.20012903  0.80034462]
 [ 0.56239043 -1.26270855]
 [ 0.63773861  0.46236393]]
The new point [[0.906 0.606]] belongs to cluster: 1
```

Explanation:

- The centroids of the 3 clusters are shown in the normalized coordinate space.
- The algorithm predicts that the new point `VAR1 = 0.906` and `VAR2 = 0.606` belongs to cluster 1, based on its proximity to the centroids.
- This classification helps determine which group the new data point is most likely to belong to.
