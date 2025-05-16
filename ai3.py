
from typing import Callable, List, Tuple
import numpy as np
import numpy.random as ran
import gzip
import pickle
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_value: np.ndarray) -> np.ndarray:
    """Takes the sigmoid output directly and computes derivative"""
    return sigmoid_value * (1 - sigmoid_value)

# Loss functions
def sq_error(output: np.ndarray, expected: np.ndarray) -> float:
    diff = output - expected
    return np.mean(np.sum(diff**2, axis=0))

def sq_error_derivative(output: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return 2 * (output - expected) / output.shape[1]  # Divide by batch size

# Type definitions
ArrayFunction = Callable[[np.ndarray], np.ndarray]
ErrorFunction = Callable[[np.ndarray, np.ndarray], float]
ErrorFunctionDerivative = Callable[[np.ndarray, np.ndarray], np.ndarray]

class NeuralNetwork:
    def __init__(self, 
                 layers_dim: List[int],
                 non_linear_func: ArrayFunction = sigmoid,
                 non_linear_func_derivative: ArrayFunction = sigmoid_derivative,
                 error_function: ErrorFunction = sq_error,
                 error_function_derivative: ErrorFunctionDerivative = sq_error_derivative):
        
        self.layers_dim = layers_dim
        # Initialize weights with Xavier/Glorot initialization for better convergence
        self.weights = [ran.randn(y, x) * np.sqrt(1/x) for x, y in zip(layers_dim, layers_dim[1:])]
        self.biases = [ran.randn(y, 1) for y in layers_dim[1:]]
        self.activations = [None] * len(layers_dim)  # Store activations at each layer
        
        self.non_linear_func = non_linear_func
        self.non_linear_func_derivative = non_linear_func_derivative
        self.error_function = error_function
        self.error_function_derivative = error_function_derivative
    
    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform forward pass with batch support
        
        Args:
            input_data: shape (input_dim, batch_size)
            
        Returns:
            Output activations: shape (output_dim, batch_size)
        """
        # Store input activations
        self.activations[0] = input_data
        
        # Forward propagation through the network
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Calculate linear combination and apply activation function
            z = np.dot(w, self.activations[i]) + b
            self.activations[i+1] = self.non_linear_func(z)
        
        return self.activations[-1]
    
    def backward_pass(self, expected: np.ndarray, learning_rate: float) -> float:
        """
        Perform backward pass with batch support
        
        Args:
            expected: shape (output_dim, batch_size)
            learning_rate: learning rate for gradient descent
            
        Returns:
            Current error after update
        """
        batch_size = expected.shape[1]
        
        # Calculate initial error
        initial_error = self.error_function(self.activations[-1], expected)
        
        # Initialize gradients for output layer - using the final activation directly
        delta = self.error_function_derivative(self.activations[-1], expected) * \
                self.non_linear_func_derivative(self.activations[-1])
        
        # Backpropagate through the network
        for i in reversed(range(len(self.weights))):
            # Calculate weight and bias gradients
            dw = np.dot(delta, self.activations[i].T)
            db = np.sum(delta, axis=1, keepdims=True)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
            # Calculate delta for next layer (if not at input layer)
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * \
                        self.non_linear_func_derivative(self.activations[i])
        
        # Return current error after update
        return initial_error
    
    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray, learning_rate: float) -> float:
        """
        Train network on a single batch
        
        Args:
            X_batch: Input data with shape (input_dim, batch_size)
            y_batch: Expected outputs with shape (output_dim, batch_size)
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Error on the batch
        """
        # Forward pass
        self.forward_pass(X_batch)
        
        # Backward pass
        return self.backward_pass(y_batch, learning_rate)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              batch_size: int, epochs: int, learning_rate: float,
              X_val: np.ndarray = None, y_val: np.ndarray = None, 
              verbose: bool = True) -> dict:
        """
        Train the neural network using mini-batch gradient descent
        
        Args:
            X_train: Training data with shape (input_dim, n_samples)
            y_train: Training labels with shape (output_dim, n_samples)
            batch_size: Size of mini-batches
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            X_val: Validation data
            y_val: Validation labels
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing training history
        """
        n_samples = X_train.shape[1]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        history = {
            'train_loss': [],
            'val_loss': [] if X_val is not None else None
        }
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                batch_loss = self.train_batch(X_batch, y_batch, learning_rate)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
            
            # Store training loss
            history['train_loss'].append(epoch_loss)
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_pred = self.forward_pass(X_val)
                val_loss = self.error_function(val_pred, y_val)
                history['val_loss'].append(val_loss)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f} - val_loss: {val_loss:.6f}")
            elif verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input data
        
        Args:
            X: Input data with shape (input_dim, n_samples)
            
        Returns:
            Predictions with shape (output_dim, n_samples)
        """
        return self.forward_pass(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model performance
        
        Args:
            X: Input data with shape (input_dim, n_samples)
            y: Ground truth labels with shape (output_dim, n_samples)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        predictions = self.predict(X)
        loss = self.error_function(predictions, y)
        
        # Calculate accuracy for classification
        pred_labels = np.argmax(predictions, axis=0)
        true_labels = np.argmax(y, axis=0)
        accuracy = np.mean(pred_labels == true_labels)
        
        return loss, accuracy


def load_mnist(filename):
    """Load MNIST dataset from pickle file"""
    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    
    # Extract data
    X_train, y_train = train_set
    X_val, y_val = valid_set
    X_test, y_test = test_set
    
    # Reshape and normalize X data
    X_train = X_train.T / 255.0  # Shape: (784, n_samples)
    X_val = X_val.T / 255.0
    X_test = X_test.T / 255.0
    
    # One-hot encode y data
    def one_hot_encode(y, num_classes=10):
        y_one_hot = np.zeros((num_classes, len(y)))
        for i, label in enumerate(y):
            y_one_hot[label, i] = 1
        return y_one_hot
    
    y_train = one_hot_encode(y_train)  # Shape: (10, n_samples)
    y_val = one_hot_encode(y_val)
    y_test = one_hot_encode(y_test)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def visualize_mnist(X, y, num_examples=5):
    """Visualize MNIST examples"""
    plt.figure(figsize=(15, 3))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i+1)
        plt.imshow(X[:, i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {np.argmax(y[:, i])}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if history['val_loss'] is not None:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist('mnist.pkl.gz')
    
    # Visualize examples
    print("Dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Sample visualization
    visualize_mnist(X_train, y_train)
    
    # Create and train neural network
    print("Creating neural network...")
    
    # Network architecture: 784 (input) -> 128 (hidden) -> 64 (hidden) -> 10 (output)
    nn = NeuralNetwork([784, 128, 64, 10])
    
    print("Training neural network...")
    # Train with smaller subset for demonstration
    # Adjust the parameters for full training
    history = nn.train(
        X_train[:, :10000],
        y_train[:, :10000],
        batch_size=64,
        epochs=10,
        learning_rate=0.1,
        X_val=X_val,
        y_val=y_val
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_loss, test_accuracy = nn.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.4f}")
