
from typing import Callable, List
import numpy as np
import numpy.random as ran
import gzip
import pickle

# Original functions remain unchanged
def sigmoid(x:np.ndarray):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_value:np.ndarray):
    return sigmoid_value*(1-sigmoid_value)

def sq_error(output:np.ndarray,expected:np.ndarray)->float:
    diff = output-expected
    # Updated to handle batches by taking mean across samples
    return np.mean(np.sum(diff**2, axis=0))
    ## This is equivalent to (x1**2+x2**2+...)/n

def sq_error_derivative(output:np.ndarray,expected:np.ndarray)->np.ndarray:
    # Updated to handle batches by including the batch size in normalization
    return 2*(output-expected) / output.shape[1]
    # x axis is down ,and y axis is right
    # output.shape[1] means y axis.When you run the entire batch in there
    # and average it for average error.




ArrayFunction=Callable[[np.ndarray],np.ndarray]
ErrorFunction=Callable[[np.ndarray,np.ndarray],float]
ErrorFunctionDerivative=Callable[[np.ndarray,np.ndarray],np.ndarray]

class NeuralNetwork:
    def __init__(self,layers_dim:List[int],
                 non_linear_func:ArrayFunction=sigmoid,
                 non_linear_func_derivitave:ArrayFunction=sigmoid_derivative,
                 errorFunction:ErrorFunction=sq_error,
                 errorFuntionDerivative:ErrorFunctionDerivative=sq_error_derivative):
        self.layers_dim=layers_dim
        # Initialize weights and biases as before
        self.weight=[ran.randn(y,x) for x,y in zip(layers_dim,layers_dim[1:])]
        self.bias=[ran.randn(x,1) for x in layers_dim[1:]]
        # Initialize neurons (now will store activations for each batch)
        self.neuron=[np.zeros((x,1)) for x in layers_dim]
        self.non_linear_func=non_linear_func
        self.non_linear_func_derivitave=non_linear_func_derivitave
        self.errorFunction=errorFunction
        self.ErrorFunctionDerivative=errorFuntionDerivative
    
    # Updated to handle batch inputs
    def forward_pass(self, input:np.ndarray=None):
        """
        Perform forward pass with batch support
        
        Args:
            input: shape (input_dim, batch_size) - if None, use existing self.neuron[0]
        """
        if input is not None:
            # Update to handle batches - input is now a matrix where each column is a sample
            self.neuron[0] = input
        
        # Forward propagation through the network
        for i, (weight, bias, neuron, neuron1) in enumerate(zip(self.weight, self.bias, self.neuron, self.neuron[1:])):
            # For batches: perform matrix multiplication for all samples at once
            # weight @ neuron shape: (output_features, batch_size)
            # bias is broadcasted across batch dimension
            neuron1[...] = self.non_linear_func(weight @ neuron + bias)
            
        return self.neuron[-1]  # Return output activations
    
    # Updated to handle batch inputs
    def backward_pass(self, expected:np.ndarray, learning_rate:float):
        """
        Perform backward pass with batch support
        
        Args:
            expected: shape (output_dim, batch_size)
            learning_rate: learning rate for gradient descent
        """
        batch_size = expected.shape[1]
        
        print("Error before:", self.errorFunction(self.neuron[-1], expected))
        
        # Initial error derivative for output layer (now handles batches)
        in_derivative = self.ErrorFunctionDerivative(self.neuron[-1], expected) * \
                        self.non_linear_func_derivitave(self.neuron[-1])
        
        # Backpropagate through layers
        for w, b, n in zip(self.weight[::-1], self.bias[::-1], self.neuron[:-1][::-1]):
            # Batch version of derivatives
            b_derivative = np.sum(in_derivative, axis=1, keepdims=True)  # Sum across batch dimension
            w_derivative = in_derivative @ n.T  # Matrix multiplication handles all batch samples
            
            # Propagate error to previous layer (for all samples in batch)
            in_derivative = w.T @ in_derivative * self.non_linear_func_derivitave(n)
            
            # Update weights and biases (single update using batch average)
            w[...] += -learning_rate * w_derivative
            b[...] += -learning_rate * b_derivative
        
        # Perform forward pass to compute new error
        self.forward_pass()
        print("Error now:", self.errorFunction(self.neuron[-1], expected))
    
    # New method for batch training
    def train_batch(self, X_batch:np.ndarray, y_batch:np.ndarray, learning_rate:float):
        """
        Train on a single batch
        
        Args:
            X_batch: Input data with shape (input_dim, batch_size)
            y_batch: Expected outputs with shape (output_dim, batch_size)
            learning_rate: Learning rate for gradient descent
        """
        # Forward pass with batch
        self.forward_pass(X_batch)
        
        # Backward pass with batch
        self.backward_pass(y_batch, learning_rate)
    
    # New method for full dataset training with mini-batches
    def train(self, X_train:np.ndarray, y_train:np.ndarray, 
              batch_size:int, epochs:int, learning_rate:float, 
              verbose:bool=True):
        """
        Train the neural network using mini-batch gradient descent
        
        Args:
            X_train: Training data with shape (input_dim, n_samples)
            y_train: Training labels with shape (output_dim, n_samples)
            batch_size: Size of mini-batches
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            verbose: Whether to print progress
        """
        n_samples = X_train.shape[1]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            total_loss = 0
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                # Forward pass
                self.forward_pass(X_batch)
                
                # Compute loss for monitoring
                batch_loss = self.errorFunction(self.neuron[-1], y_batch)
                total_loss += batch_loss * (end_idx - start_idx) / n_samples
                
                # Backward pass
                self.backward_pass(y_batch, learning_rate)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss:.6f}")

# Function to load and preprocess MNIST
def load_mnist(filename):
    """Load MNIST dataset from pickle file"""
    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    
    # Extract data
    X_train, y_train = train_set
    X_val, y_val = valid_set
    X_test, y_test = test_set
    
    # Reshape X data to be (784, n_samples) instead of (n_samples, 784)
    # This makes matrix operations in our network more intuitive
    X_train = X_train.T / 255.0  # Also normalize to [0,1]
    X_val = X_val.T / 255.0
    X_test = X_test.T / 255.0
    
    # One-hot encode y data for multi-class classification
    def one_hot_encode(y, num_classes=10):
        y_one_hot = np.zeros((num_classes, len(y)))
        for i, label in enumerate(y):
            y_one_hot[label, i] = 1
        return y_one_hot
    
    y_train = one_hot_encode(y_train)  # Shape: (10, n_samples)
    y_val = one_hot_encode(y_val)
    y_test = one_hot_encode(y_test)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Example usage
if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist('mnist.pkl.gz')
    
    print("Dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Create neural network: 784 inputs, 128 hidden, 10 outputs
    print("Creating neural network...")
    nn = NeuralNetwork([784, 128, 10])
    
    # Train with mini-batches
    print("Training neural network...")
    nn.train(
        X_train[:, :10000],  # Using first 10000 samples for faster training
        y_train[:, :10000],
        batch_size=64,
        epochs=5,
        learning_rate=0.1
    )
    
    # Evaluate on some test samples
    print("Evaluating on test samples...")
    nn.forward_pass(X_test[:, :100])  # First 100 test samples
    predictions = np.argmax(nn.neuron[-1], axis=0)
    true_labels = np.argmax(y_test[:, :100], axis=0)
    accuracy = np.mean(predictions == true_labels)
    print(f"Test accuracy on 100 samples: {accuracy:.4f}")
