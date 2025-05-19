import numpy as np

import pickle
import gzip
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

if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist('./data/mnist.pkl.gz')
    
    print("Dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, yVal: {y_val.shape}")
    print(f"Y_train: {list(y_train.argmax(axis=0))}") 
