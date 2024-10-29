import numpy as np


# Define the Sigmoid Activation Function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# XOR dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels_xor = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1

weights_input_hidden = np.random.uniform(
    size=(input_layer_neurons, hidden_layer_neurons)
)
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Training parameters
learning_rate = 0.5
epochs = 10000

# Backpropagation Training
for epoch in range(epochs):
    # Forward Pass
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = (
        np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    )
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    error = labels_xor - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating weights and biases
    weights_hidden_output += (
        hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    )
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print("Training complete for XOR using Backpropagation.")
print("Predicted Output after training:")
print(predicted_output.round())
