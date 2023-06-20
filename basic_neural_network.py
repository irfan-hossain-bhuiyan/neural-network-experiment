
import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

        # Initialize biases
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)

            # Backpropagation
            error = y - output
            delta2 = error * self.sigmoid_derivative(output)
            error_hidden = delta2.dot(self.W2.T)
            delta1 = error_hidden * self.sigmoid_derivative(self.a1)

            # Update weights and biases
            self.W2 += self.a1.T.dot(delta2) * learning_rate
            self.b2 += np.sum(delta2, axis=0, keepdims=True) * learning_rate
            self.W1 += X.T.dot(delta1) * learning_rate
            self.b1 += np.sum(delta1, axis=0, keepdims=True) * learning_rate

    def sigmoid_derivative(self, x):
        return x * (1 - x)
