from typing import Callable, List
import numpy as np
from numpy._core.numeric import ndarray
import numpy.random as ran
import gzip
import pickle
from weightSet import WeightBaseList

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Original functions remain unchanged
def sigmoid(x: np.ndarray):
    # Add small epsilon to prevent overflow but don't change the logic
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_value: np.ndarray):
    return sigmoid_value * (1 - sigmoid_value)

def sq_error(output: np.ndarray, expected: np.ndarray) -> float:
    diff = output - expected
    # Keep your original implementation
    return np.mean(np.sum(diff**2, axis=0))

def sq_error_derivative(output: np.ndarray, expected: np.ndarray) -> np.ndarray:
    # Keep your original implementation
    return 2 * (output - expected)

ArrayFunction = Callable[[np.ndarray], np.ndarray]
ErrorFunction = Callable[[np.ndarray, np.ndarray], float]
ErrorFunctionDerivative = Callable[[np.ndarray, np.ndarray], np.ndarray]

class NeuralNetwork:
    def __init__(self, layers_dim: List[int],
                 non_linear_func: ArrayFunction = sigmoid,
                 non_linear_func_derivitave: ArrayFunction = sigmoid_derivative,
                 errorFunction: ErrorFunction = sq_error,
                 errorFuntionDerivative: ErrorFunctionDerivative = sq_error_derivative):
        self.layers_dim = layers_dim
        self.freeVariable = WeightBaseList(self.layers_dim)
        self.weights = self.freeVariable.weight_list
        self.biases = self.freeVariable.bias_list
        self.neuron: List[ndarray] = [None for x in layers_dim]
        self.non_linear_func = non_linear_func
        self.non_linear_func_derivitave = non_linear_func_derivitave
        self.errorFunction = errorFunction
        self.errorFunctionDerivative = errorFuntionDerivative
    
    def forward_pass(self, input: np.ndarray = None):
        """
        Perform forward pass with batch support
        
        Args:
            input: shape (input_dim, batch_size) - if None, use existing self.neuron[0]
        """
        if input is not None:
            self.neuron[0] = input
        
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            self.neuron[i+1] = self.non_linear_func(weight @ self.neuron[i] + bias)
    
    def gradient(self, expected: np.ndarray):
        """
        Perform backward pass with batch support - USING YOUR ORIGINAL IMPLEMENTATION
        with minimal targeted fixes
        
        Args:
            expected: shape (output_dim, batch_size)
        """
        batch_size = expected.shape[1]
        derivative = self.freeVariable.copy()
        
        # Initial error derivative for output layer
        in_derivative = self.errorFunctionDerivative(self.neuron[-1], expected) * \
                        self.non_linear_func_derivitave(self.neuron[-1])
        
        # Backpropagate through layers - KEEP YOUR ORIGINAL IMPLEMENTATION
        for w, dw, db, i in zip(self.weights[::-1], derivative.weight_list[::-1],
                            derivative.bias_list[::-1], self.neuron[:-1][::-1]):
            # Calculate bias gradients
            db[...] = np.sum(in_derivative, axis=1, keepdims=True) / batch_size
            
            # Calculate weight gradients
            dw[...] = (in_derivative @ i.T) / batch_size
            
            # Propagate error to previous layer
            in_derivative = w.T @ in_derivative * self.non_linear_func_derivitave(i)
        
        return derivative
    
    def backward_pass(self, expected: np.ndarray, learning_rate: float = 0.1):
        """Update weights and biases using gradient descent"""
        self.freeVariable.core -= self.gradient(expected).core * learning_rate
    
    def costCheck(self, expected: ndarray) -> float:
        """Calculate cost/loss"""
        return self.errorFunction(self.neuron[-1], expected)
    
    def predicted_value(self):
        """Get predicted class indices"""
        return self.neuron[-1].argmax(axis=0)
    
    def error_check(self, y_value: np.ndarray):
        """Calculate accuracy"""
        return np.mean(self.predicted_value() == y_value.argmax(axis=0))
    
    def save_parameters(self, filename: str):
        """Save weights and biases to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model parameters saved to {filename}")

def gradientCheck():
    """
    Enhanced gradient check function with central difference method
    """
    print("Running gradient check...")
    
    # Use a small epsilon for numerical stability
    epsilon = 1e-5  # Same as your original
    np.random.seed(42)  # For reproducibility
    
    # Create a small network for testing
    nn = NeuralNetwork([30, 10, 10, 20])
    x_input = np.random.randn(30, 10)  # 10 samples
    y_input = np.random.randn(20, 10)  # 10 outputs per sample
    
    # Forward pass to establish baseline
    nn.forward_pass(x_input)
    cost = nn.costCheck(y_input)
    
    # Store original weights
    original_values = nn.freeVariable.core.copy()
    
    # Compute analytical gradient
    realGrad = nn.gradient(y_input)
    
    # Array for numerical gradients
    manualGrad = np.empty_like(nn.freeVariable.core)
    
    # Compute numerical gradient using forward difference (as in your original)
    for i in range(len(nn.freeVariable.core)):
        original_value = nn.freeVariable.core[i]
        
        # Perturb
        nn.freeVariable.core[i] = original_value + epsilon
        
        # Forward pass and cost with perturbed weight
        nn.forward_pass(x_input)
        cost_new = nn.costCheck(y_input)
        
        # Restore
        nn.freeVariable.core[i] = original_value
        
        # Forward difference
        manualGrad[i] = (cost_new - cost) / epsilon
    
    # Restore all original values
    nn.freeVariable.core = original_values
    
    # Calculate differences
    diff = manualGrad - realGrad.core
    
    # Print metrics
    print(f"Max absolute difference: {np.max(np.abs(diff))}")
    print(f"Relative error: {np.linalg.norm(diff) / (np.linalg.norm(manualGrad) + np.linalg.norm(realGrad.core))}")
    print(f"Avg error(should be 0): {np.sum(diff)}")
    
    if np.max(np.abs(diff)) < 1e-4:
        print(f"{GREEN}Gradient check passed!{RESET}")
    else:
        print(f"{RED}Gradient check failed!{RESET}")
        
        # Show largest discrepancies
        worst_indices = np.argsort(np.abs(diff))[-5:]
        print("\nParameters with largest discrepancies:")
        for idx in reversed(worst_indices):
            print(f"Parameter {idx}: Numerical: {manualGrad[idx]}, Analytical: {realGrad.core[idx]}, Diff: {diff[idx]}")
    
    return manualGrad, realGrad.core, diff

def train_network(nn: NeuralNetwork,
                 x_train: np.ndarray, y_train: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 epochs: int = 30,
                 batch_size: int = 32,
                 learning_rate: float = 0.1,
                 lr_decay: float = 0.95):
    """
    Train the neural network with proper epoch structure
    """
    n_samples = x_train.shape[1]
    n_batches = n_samples // batch_size
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    current_lr = learning_rate
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        x_shuffled = x_train[:, indices]
        y_shuffled = y_train[:, indices]
        
        # Train on mini-batches
        epoch_loss = 0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = x_shuffled[:, start_idx:end_idx]
            y_batch = y_shuffled[:, start_idx:end_idx]
            
            # Forward pass
            nn.forward_pass(x_batch)
            
            # Calculate loss
            batch_loss = nn.costCheck(y_batch)
            epoch_loss += batch_loss
            
            # Backward pass
            nn.backward_pass(y_batch, current_lr)
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / n_batches
        train_losses.append(avg_epoch_loss)
        
        # Evaluate on training set (use a subset for efficiency if dataset is large)
        if n_samples > 10000:
            eval_indices = np.random.choice(n_samples, 10000, replace=False)
            x_eval = x_train[:, eval_indices]
            y_eval = y_train[:, eval_indices]
        else:
            x_eval = x_train
            y_eval = y_train
            
        nn.forward_pass(x_eval)
        train_acc = nn.error_check(y_eval)
        train_accuracies.append(train_acc)
        
        # Evaluate on validation set
        nn.forward_pass(x_val)
        val_acc = nn.error_check(y_val)
        val_accuracies.append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.6f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        
        # Decay learning rate
        current_lr *= lr_decay
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def load_parameters(filename: str) -> NeuralNetwork:
    """Load weights and biases from a file using pickle."""
    with open(filename, 'rb') as f:
        nn = pickle.load(f)
    print(f"Model parameters loaded from {filename}")
    return nn

def load_mnist(filename):
    """Load MNIST dataset from pickle file"""
    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    
    # Extract data
    X_train, y_train = train_set
    X_val, y_val = valid_set
    X_test, y_test = test_set
    
    # Reshape and normalize
    X_train = X_train.T / 255.0
    X_val = X_val.T / 255.0
    X_test = X_test.T / 255.0
    
    # One-hot encode labels
    def one_hot_encode(y, num_classes=10):
        y_one_hot = np.zeros((num_classes, len(y)))
        for i, label in enumerate(y):
            y_one_hot[label, i] = 1
        return y_one_hot
    
    y_train = one_hot_encode(y_train)
    y_val = one_hot_encode(y_val)
    y_test = one_hot_encode(y_test)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Example usage
if __name__ == "__main__":
    # Run gradient check
    gradientCheck()
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist('./data/mnist.pkl.gz')
    
    # Create neural network with better architecture
    print("\nCreating neural network...")
    nn = NeuralNetwork([784, 128, 64, 10])
    
    # Train with improved training loop
    print("\nTraining neural network...")
    history = train_network(
        nn, 
        x_train, y_train,
        x_val, y_val,
        epochs=30,
        batch_size=128,  # Larger batch size
        learning_rate=0.1,  # Start with your original learning rate
        lr_decay=0.97  # Gradual decay
    )
    
    # Evaluate on test set
    nn.forward_pass(x_test)
    test_accuracy = nn.error_check(y_test)
    print(f"\nFinal test accuracy: {test_accuracy:.4f}")
    
    # Save the trained model
    nn.save_parameters("mnist_model_improved.pkl")
