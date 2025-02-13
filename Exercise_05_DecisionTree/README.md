# Exercise 05

## Implement Naïve Bayes Theorem to Classify English Text

### Aim  

To implement the Naïve Bayes theorem for text classification using Python.

### Theory

Naïve Bayes is a probabilistic classifier based on Bayes' theorem, which assumes that the features (words in this case) are independent of each other. The classifier calculates the probability of a class given a set of features (text), and the class with the highest probability is selected.

The classification rule follows:

$$
P(\text{Class}|\text{Text}) = \frac{P(\text{Text}|\text{Class}) \cdot P(\text{Class})}{P(\text{Text})}
$$

Where:  

- $P(\text{Class}|\text{Text})$ is the probability of the class given the text.  
- $P(\text{Text}|\text{Class})$ is the likelihood of the text given the class.  
- $P(\text{Class})$ is the prior probability of the class.  
- $P(\text{Text})$ is the probability of the text.

Naïve Bayes works by calculating the posterior probability for each class and selecting the class with the highest probability.

### Procedure/Program  

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# load the 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')

X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.3, random_state=42)

# convert text data to numerical vectors using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)
y_pred = nb_classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Naïve Bayes classifier: {accuracy * 100:.2f}%")
```

### Output/Explanation  

Output:  

The program outputs:  

```bash
Accuracy of the Naïve Bayes classifier: 87.30%
```

Explanation:  

- The 20 newsgroups dataset is loaded, containing text data categorized into multiple topics.  
- The dataset is split into training and test sets.  
- The `CountVectorizer` is used to convert the text data into numerical feature vectors by counting the occurrences of words, while ignoring common stopwords.  
- A `MultinomialNB` classifier is used to train the model on the training data.  
- The model then makes predictions on the test data, and the accuracy of the model is computed and displayed.
