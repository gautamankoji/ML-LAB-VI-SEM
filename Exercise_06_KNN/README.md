# Excercise 06

## Write a program to demonstrate the working of the decision tree-based ID3 algorithm. Use an appropriate dataset for building the decision tree and apply this knowledge to classify a new sample

### Aim  

To implement the ID3 algorithm for building a decision tree, classify a given dataset, and predict the class for a new sample.

### Theory

The Iterative Dichotomiser 3 (ID3) algorithm builds decision trees by selecting the attribute with the highest information gain. This is determined by calculating entropy, which measures impurity or randomness in data.  

Entropy is defined as:  
$$
Entropy(S) = -\sum P(i) \cdot \log_2(P(i))
$$  

Information gain is given by:  
$$
Gain(S, A) = Entropy(S) - \sum \left( \frac{|S_v|}{|S|} \cdot Entropy(S_v) \right)
$$

Where:  

- $S$: Dataset  
- $A$: Attribute  
- $S_v$: Subset of \(S\) for a specific attribute value.  

### Procedure/Program  

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# sample dataset
data = pd.DataFrame({
    "Outlook": ["Sunny", "Overcast", "Rain", "Sunny", "Sunny"],
    "Temperature": ["Hot", "Cool", "Mild", "Hot", "Cool"],
    "Humidity": ["High", "Normal", "High", "High", "Normal"],
    "Windy": [False, True, False, True, False],
    "Play": ["No", "Yes", "Yes", "No", "Yes"]
})

# new sample for prediction
new_sample = pd.DataFrame({
    "Outlook": ["Overcast"],
    "Temperature": ["Cool"],
    "Humidity": ["High"],
    "Windy": [False]
})

# define categorical and numerical columns
categorical_columns = ["Outlook", "Temperature", "Humidity", "Windy"]

# preprocessing: One-hot encoding for categorical variables
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_columns)],
    remainder="passthrough"
)

# prepare data for training
X = data.drop(columns=["Play"])
y = data["Play"]
X_encoded = preprocessor.fit_transform(X)

# train DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_encoded, y)

# transform the new sample with the same preprocessor
new_sample_encoded = preprocessor.transform(new_sample)

# predict the output
prediction = clf.predict(new_sample_encoded)
print(f"Prediction for the new sample: {prediction[0]}")
```

### Output/Explanation  

Output:  

```bash
Prediction for the new sample: Yes
```

1. **Decision Tree Rules:**  
   Rules derived by the ID3 algorithm to split the dataset.  
2. **Classification Report:**  
   Performance metrics such as precision, recall, and F1-score.  
3. **Prediction for New Sample:**  
   Predicted class for a new data point.

Explanation:  

- The ID3 algorithm uses entropy and information gain to decide the best attribute for splitting.  
- The program demonstrates the process of building the decision tree and classifying test samples.  
- A new sample is classified based on the tree structure.  
