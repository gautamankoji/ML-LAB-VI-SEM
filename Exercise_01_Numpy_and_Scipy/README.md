# Exercise 01

## Study of Python Basic Libraries such as NumPy and SciPy  

### Aim  

To understand and experiment with basic functionalities of Python libraries NumPy and SciPy.  

### Procedure/Program  

#### NumPy Example  

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("NumPy Array:\n", arr)

# basic operations
print("Shape of array:", arr.shape)
print("Sum of all elements:", np.sum(arr))
print("Mean of elements:", np.mean(arr))
print("Transpose of array:\n", arr.T)
```

#### SciPy Example  

```python
from scipy import stats
import numpy as np

data = [12, 15, 14, 10, 18, 14, 17, 16, 15, 15]

mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode.mode)
```

### Output/Explanation  

- **Output:**  

  The program demonstrates basic NumPy and SciPy functionalities, with an expected output similar to:  

  ```bash
  NumPy Array:
   [[1 2 3]
   [4 5 6]]
  Shape of array: (2, 3)
  Sum of all elements: 21
  Mean of elements: 3.5
  Transpose of array:
   [[1 4]
   [2 5]
   [3 6]]
  
  Mean: 14.6
  Median: 15.0
  Mode: 15
  ```

- **Explanation:**  
  - NumPy is used for array creation, mathematical operations, and reshaping.  
  - SciPy's `stats` module provides statistical functions such as mean, median, and mode.  
  - The program showcases how these libraries simplify numerical computations.
