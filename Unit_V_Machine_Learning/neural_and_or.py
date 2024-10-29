import numpy as np


# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Training dataset for AND and OR gates
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Labels for AND and OR gates
labels_and = np.array([[0], [0], [0], [1]])  # AND gate
labels_or = np.array([[0], [1], [1], [1]])  # OR gate

# Initialize weights and bias randomly
weights = np.random.rand(2, 1)
bias = np.random.rand(1)
learning_rate = 0.1
epochs = 10000


# Function to train the network
def train_gate(inputs, labels):
    global weights, bias
    for epoch in range(epochs):
        # Forward pass
        z = np.dot(inputs, weights) + bias
        outputs = sigmoid(z)

        # Calculate error
        error = labels - outputs
        adjustments = error * sigmoid_derivative(outputs)

        # Update weights and bias
        weights += np.dot(inputs.T, adjustments) * learning_rate
        bias += np.sum(adjustments) * learning_rate

    return outputs


# Train and test AND gate
print("Training for AND Gate")
train_gate(inputs, labels_and)
print("Predictions for AND Gate:", sigmoid(np.dot(inputs, weights) + bias).round())

# Train and test OR gate
print("\nTraining for OR Gate")
weights = np.random.rand(2, 1)  # Re-initialize weights for OR
bias = np.random.rand(1)
train_gate(inputs, labels_or)
print("Predictions for OR Gate:", sigmoid(np.dot(inputs, weights) + bias).round())
