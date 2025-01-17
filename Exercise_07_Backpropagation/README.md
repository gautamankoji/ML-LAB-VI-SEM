# Exercise 07

## Build an Artificial Neural Network by implementing the Backpropagation algorithm and test the same using appropriate datasets

### Aim  

To build an Artificial Neural Network (ANN) by implementing the Backpropagation algorithm and test its performance using an appropriate dataset.

### Theory

Artificial Neural Networks (ANNs) are computational models inspired by biological neural networks. They consist of layers of interconnected neurons, where:  

- **Input Layer:** Accepts input features.  
- **Hidden Layers:** Processes data with weights, biases, and activation functions.  
- **Output Layer:** Produces predictions.  

**Backpropagation Algorithm:**  
Backpropagation is used to minimize the error by updating weights and biases through gradient descent.  

Steps:  

1. **Forward Propagation:** Compute outputs.  
2. **Compute Loss:** Measure the difference between actual and predicted values.  
3. **Backward Propagation:** Calculate gradients of the loss with respect to weights.  
4. **Update Weights:** Adjust weights using the gradients to minimize loss.  

Mathematically:  
$$
\Delta w = -\eta \frac{\partial L}{\partial w}
$$  
Where:  

- $\eta$: Learning rate  
- $L$: Loss function  

### Procedure/Program

```python
import numpy as np

# activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# initialize weights and biases
np.random.seed(42)
input_neurons = X.shape[1]
hidden_neurons = 4
output_neurons = 1

weights_input_hidden = np.random.uniform(-1, 1, size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(-1, 1, size=(hidden_neurons, output_neurons))
bias_hidden = np.random.uniform(-1, 1, size=(1, hidden_neurons))
bias_output = np.random.uniform(-1, 1, size=(1, output_neurons))
learning_rate = 0.1

# train the ANN
epochs = 10000
for epoch in range(epochs):
    # forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = tanh(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # compute loss
    error = y - predicted_output
    loss = np.mean(error ** 2)

    # backward propagation
    d_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * tanh_derivative(hidden_layer_output)

    # update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # print loss at intervals
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# test the ANN
hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
hidden_layer_output = tanh(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predicted_output = sigmoid(output_layer_input)
predicted_classes = (predicted_output > 0.5).astype(int)

# evaluate
accuracy = np.mean(predicted_classes == y)
print(f"Accuracy on XOR problem: {accuracy * 100:.2f}%")
```

### Output/Explanation  

Output:

```bash
Epoch 0, Loss: 0.268363
Epoch 1000, Loss: 0.011625
Epoch 2000, Loss: 0.003529
Epoch 3000, Loss: 0.001965
Epoch 4000, Loss: 0.001336
Epoch 5000, Loss: 0.001003
Epoch 6000, Loss: 0.000798
Epoch 7000, Loss: 0.000661
Epoch 8000, Loss: 0.000562
Epoch 9000, Loss: 0.000489
Accuracy on XOR problem: 100.00%
```

1. **Training Loss:** Displayed at intervals to show model improvement.  
2. **Accuracy:** Final accuracy of the model on test data.  

Explanation:

- The ANN learns to solve the XOR problem through forward and backward propagation.  
- Weights are updated iteratively using gradients to minimize the loss.  
- The trained model is evaluated on test data to measure performance.  

This implementation demonstrates the working of a simple neural network with backpropagation using Python.
