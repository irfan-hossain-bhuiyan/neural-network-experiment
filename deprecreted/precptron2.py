
from typing import Callable, List
import numpy as np
from numpy._core.numeric import ndarray
import numpy.random as ran
import gzip
import pickle

# Improved sigmoid function with numerical stability
def sigmoid(x:np.ndarray):
    # Clip values to avoid overflow
    x_safe = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_safe))

def sigmoid_derivative(sigmoid_value:np.ndarray):
    return sigmoid_value*(1-sigmoid_value)

def sq_error(output:np.ndarray, expected:np.ndarray)->float:
    diff = output-expected
    # Handle batches by taking mean across samples
    return np.mean(np.sum(diff**2, axis=0))

def sq_error_derivative(output:np.ndarray, expected:np.ndarray)->np.ndarray:
    # Handle batches by including the batch size in normalization
    return 2*(output-expected) / output.shape[1]

ArrayFunction=Callable[[np.ndarray],np.ndarray]
ErrorFunction=Callable[[np.ndarray,np.ndarray],float]
ErrorFunctionDerivative=Callable[[np.ndarray,np.ndarray],np.ndarray]

class NeuralNetwork:
    def __init__(self, layers_dim:List[int],
                 non_linear_func:ArrayFunction=sigmoid,
                 non_linear_func_derivitave:ArrayFunction=sigmoid_derivative,
                 errorFunction:ErrorFunction=sq_error,
                 errorFuntionDerivative:ErrorFunctionDerivative=sq_error_derivative):
        self.layers_dim=layers_dim
        # Initialize weights using He initialization for better convergence
        #self.weights=[ran.randn(y, x) * np.sqrt(2/x) for x, y in zip(layers_dim, layers_dim[1:])]
        total_weight=sum([n*n1 for n,n1 in zip(self.layers_dim,self.layers_dim[1:])])
        total_bias=sum(self.layers_dim[1:])
        self.weight=np.random.rand(total_weight)*2-1
        self.bias=np.random.rand(total_bias)*2-1
        previous=0
        self.weight_list=[]
        self.bias_list=[]
        for n,n1 in zip(self.layers_dim,self.layers_dim[1:]):
            self.weight_list.append(self.weight[previous:previous+n*n1].reshape((n1,n)))
            previous+=n*n1
        previous=0
        for n in self.layers_dim[1:]:
            self.bias_list.append(self.bias[previous:previous+n].reshape((n,1)))
            previous+=n
        self.bias_list=[np.zeros((x, 1)) for x in layers_dim[1:]]  # Initialize biases to zeros
        self.neuron:List[ndarray]=[None for x in layers_dim]
        self.non_linear_func=non_linear_func
        self.non_linear_func_derivitave=non_linear_func_derivitave
        self.errorFunction=errorFunction
        self.errorFunctionDerivative=errorFuntionDerivative
    
    def forward_pass(self, input:np.ndarray=None):
        """
        Perform forward pass with batch support
        
        Args:
            input: shape (input_dim, batch_size) - if None, use existing self.neuron[0]
        """
        if input is not None:
            self.neuron[0] = input
        
        for i, (weight, bias) in enumerate(zip(self.weight_list, self.bias_list)):
            # Broadcasting bias across batch dimension
            self.neuron[i+1] = self.non_linear_func(weight @ self.neuron[i] + bias)
        return self.neuron[-1]
    
    def backward_pass(self, expected:np.ndarray, learning_rate:float):
        """
        Perform backward pass with batch support
        
        Args:
            expected: shape (output_dim, batch_size)
            learning_rate: learning rate for gradient descent
        
        Returns:
            error: current error after parameter update
        """
        batch_size = expected.shape[1]
        
        # Get initial error - but don't assert it will decrease
        errorBefore = self.errorFunction(self.neuron[-1], expected)
        
        # Initial error derivative for output layer
        in_derivative = self.errorFunctionDerivative(self.neuron[-1], expected) * \
                        self.non_linear_func_derivitave(self.neuron[-1])
        
        # Backpropagate through layers
        for w, b, n in zip(self.weight_list[::-1], self.bias_list[::-1], self.neuron[:-1][::-1]):
            # Average gradients across batch dimension
            b_derivative = np.mean(in_derivative, axis=1, keepdims=True)
            # Compute weight gradients (average across batch)
            w_derivative = np.dot(in_derivative,n)/batch_size
            
            # Propagate error to previous layer
            in_derivative = w.T @ in_derivative * self.non_linear_func_derivitave(n)
            
            # Update weights and biases using gradient descent
            w[...] -= learning_rate * w_derivative 
            b[...] -= learning_rate * b_derivative 
        
        # Perform forward pass to compute new error
        self.forward_pass()
        errorNow = self.errorFunction(self.neuron[-1], expected)
        
        # Return error instead of asserting - this lets the caller monitor progress
        return errorNow
    
    def train_batch(self, X_batch:np.ndarray, y_batch:np.ndarray, learning_rate:float):
        """
        Train on a single batch
        
        Args:
            X_batch: Input data with shape (input_dim, batch_size)
            y_batch: Expected outputs with shape (output_dim, batch_size)
            learning_rate: Learning rate for gradient descent
            
        Returns:
            batch_error: Error after training on this batch
        """
        # Forward pass with batch
        self.forward_pass(X_batch)
        
        # Backward pass with batch
        batch_error = self.backward_pass(y_batch, learning_rate)
        return batch_error
    
    def train(self, X_train:np.ndarray, y_train:np.ndarray, 
              batch_size:int, epochs:int, learning_rate:float,
              X_val:np.ndarray=None, y_val:np.ndarray=None,
              verbose:bool=True):
        """
        Train the neural network using mini-batch gradient descent
        
        Args:
            X_train: Training data with shape (input_dim, n_samples)
            y_train: Training labels with shape (output_dim, n_samples)
            batch_size: Size of mini-batches
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print progress
        
        Returns:
            history: Dictionary containing training and validation losses
        """
        n_samples = X_train.shape[1]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # For tracking training progress
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Adaptive learning rate - reduces learning rate if no improvement for several epochs
        patience = 5
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            epoch_losses = []
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                # Train on batch and record loss
                batch_loss = self.train_batch(X_batch, y_batch, learning_rate)
                epoch_losses.append(batch_loss)
            
            # Average loss for this epoch
            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validate if validation data is provided
            val_loss, val_accuracy = None, None
            if X_val is not None and y_val is not None:
                val_loss, val_accuracy = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                # Learning rate scheduling based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    learning_rate *= 0.5  # Reduce learning rate
                    patience_counter = 0
                    print(f"Reducing learning rate to {learning_rate}")
            
            if verbose:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.6f} - val_loss: {val_loss:.6f} - val_acc: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.6f}")
                    
        return history
    
    def evaluate(self, X:np.ndarray, y:np.ndarray):
        """
        Evaluate model on data
        
        Args:
            X: Input data with shape (input_dim, n_samples)
            y: Ground truth labels with shape (output_dim, n_samples)
            
        Returns:
            loss: Average loss on data
            accuracy: Classification accuracy if applicable
        """
        # Forward pass
        predictions = self.forward_pass(X)
        
        # Compute loss
        loss = self.errorFunction(predictions, y)
        
        # Compute accuracy if one-hot encoded data
        if y.shape[0] > 1:  # Multi-class classification
            predicted_classes = np.argmax(predictions, axis=0)
            true_classes = np.argmax(y, axis=0)
            accuracy = np.mean(predicted_classes == true_classes)
        else:  # Binary classification
            predicted_classes = (predictions > 0.5).astype(int)
            accuracy = np.mean(predicted_classes == y)
            
        return loss, accuracy

    def predict(self, X:np.ndarray):
        """
        Make predictions on new data
        
        Args:
            X: Input data with shape (input_dim, n_samples)
            
        Returns:
            predictions: Model predictions
        """
        return self.forward_pass(X)
    
    def save_parameters(self, filename: str):
        """Save weights and biases to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.weight_list, 'biases': self.bias_list}, f)
        print(f"Model parameters saved to {filename}")
    
    def load_parameters(self, filename: str):
        """Load weights and biases from a file using pickle."""
        with open(filename, 'rb') as f:
            params = pickle.load(f)
            self.weight_list = params['weights']
            self.bias_list = params['biases']
        print(f"Model parameters loaded from {filename}")

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
    X_train = X_train.T / 255.0  # Normalize to [0,1]
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
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist('./data/mnist.pkl.gz')
    
    print("Dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Create neural network: 784 inputs, hidden layers, 10 outputs
    print("Creating neural network...")
    nn = NeuralNetwork([784, 128, 64, 10])
    
    # Train with mini-batches and validation
    print("Training neural network...")
    history = nn.train(
        X_train,
        y_train,
        batch_size=64,
        epochs=20,
        learning_rate=0.1,
        X_val=X_val,
        y_val=y_val
    )
    
    # Save the trained model
    nn.save_parameters("mnist_model.pkl")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = nn.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.4f}")
