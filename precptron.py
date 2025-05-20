
from typing import Callable, List
import numpy as np
from numpy._core.numeric import ndarray
import numpy.random as ran
import gzip
import pickle
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
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
    return 2*(output-expected)
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
        self.weights=[ran.randn(y,x)-0.5 for x,y in zip(layers_dim,layers_dim[1:])]
        self.biases=[ran.randn(x,1)-0.5 for x in layers_dim[1:]]
        # Neuron aren't initialized as array because of batches,As their are veraity of batches
        self.neuron:List[ndarray]=[None for x in layers_dim]
        self.non_linear_func=non_linear_func
        self.non_linear_func_derivitave=non_linear_func_derivitave
        self.errorFunction=errorFunction
        self.errorFunctionDerivative=errorFuntionDerivative
    
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
        
        for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
            self.neuron[i+1]=self.non_linear_func(weight @ self.neuron[i] + bias)
        #return self.neuron[-1]  # Return output activations
    
    # Updated to handle batch inputs
    def backward_pass(self, expected:np.ndarray, learning_rate:float):
        """
        Perform backward pass with batch support
        
        Args:
            expected: shape (output_dim, batch_size)
            learning_rate: learning rate for gradient descent
        """
        batch_size = expected.shape[1]
        
        
        # Initial error derivative for output layer (now handles batches)
        in_derivative = self.errorFunctionDerivative(self.neuron[-1], expected) * \
                        self.non_linear_func_derivitave(self.neuron[-1])
        # in_derivative is now (neuron number, batch size)
        
        # Backpropagate through layers
        for w, b, i in zip(self.weights[::-1], self.biases[::-1], self.neuron[:-1][::-1]):
            # Batch version of derivatives
            b_derivative = np.sum(in_derivative, axis=1,keepdims=True)/batch_size  # Sum across batch dimension
            # not need to divide by batch size as it is mean
            """
                [[1,2,3],
                 [2,3,4],
                 [3,4,5],],This is the input derivative and having mean in axis=1 means it average in x axis [[2],
                                                                                                              [3],
                                                                                                              [4],]

            """
            w_derivative = (in_derivative @ i.T)/batch_size # Matrix multiplication handles all batch samples

            """
               w_derivative= [1   (inderivative)*[1 2 3 4](neuron)
                              2
                              3
                              4]
               With batch  [[1 5 9  13]  @ [[1 2 3 4]       
                            [2 6 10 14]     [2 3 4 5]
                            [3 7 11 15]     [6 7 8 9]
                            [4 8 12 16]]    [10 11 12 13]]

               Those things got automatically sums up
            """
            
            # Propagate error to previous layer (for all samples in batch)
            in_derivative = w.T @ in_derivative * self.non_linear_func_derivitave(i)
            
            # Update weights and biases (single update using batch average)
            w[...] -= learning_rate * w_derivative 
            b[...] -= learning_rate * b_derivative 
        
    def costCheck(self,expected:ndarray)->float:
        return self.errorFunction(self.neuron[-1],expected)
    # New method for batch training
    def train_batch(self, X_batch:np.ndarray, y_batch:np.ndarray, learning_rate:float,batch_iteration:int=1):
        """
        Train on a single batch
        
        Args:
            X_batch: Input data with shape (input_dim, batch_size)
            y_batch: Expected outputs with shape (output_dim, batch_size)
            learning_rate: Learning rate for gradient descent
        """
        # Forward pass with batch
        error=0
        for _ in range(batch_iteration):
            self.forward_pass(X_batch)
            error+=self.error_check(y_batch)
            self.backward_pass(y_batch, learning_rate)
        return error/batch_iteration
    def predicted_value(self):
        return self.neuron[-1].argmax(axis=0)
    def error_check(self,y_value:np.ndarray):
        return np.mean(self.predicted_value()==y_value.argmax(axis=0))
    # New method for full dataset training with mini-batches
    def save_parameters(self, filename: str):
        """Save weights and biases to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model parameters saved to {filename}")
    



def load_parameters(filename: str)->NeuralNetwork:
        """Load weights and biases from a file using pickle."""
        with open(filename, 'rb') as f:
            nn = pickle.load(f)
        print(f"Model parameters loaded from {filename}")
        return nn

def trainBatchall (nn:NeuralNetwork, x_train:np.ndarray, y_train:np.ndarray, 
                   batch_size:int, learning_rate:float,batchIteration:int=1):
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
        # X_train is (784,sample size)
        n_data = x_train.shape[1]
        chunkAmount = n_data//batch_size
        trainError=0
        for chunkNo in range(0,chunkAmount):
            chunkIndex=chunkNo*chunkAmount
            chunkIndex1=chunkIndex+batch_size
            trainError+=nn.train_batch(X_batch=x_train[:,chunkIndex:chunkIndex1],
                           y_batch=y_train[:,chunkIndex:chunkIndex1],
                           learning_rate=learning_rate,
                           batchIteration=batchIteration,)
        trainError/=chunkAmount
        return trainError
def trainRandomBatchAll(nn:NeuralNetwork, x_train:np.ndarray, y_train:np.ndarray,
                        batch_size:int, learning_rate:float,batchIteration:int=1):
    n_data= x_train.shape[1]
    n_permutation= np.random.permutation(n_data)
    return trainBatchall(nn,x_train[:,n_permutation],y_train[:,n_permutation],batch_size,learning_rate,batchIteration)

def binaryIteration(n:int):
    yield (0,1)
    for x  in range(1,n):
        i=0
        axis=x
        while axis%2==0:
            i+=1
            axis>>=1
            xm1=x-(1<<i)
            yield (xm1,x)
        yield (x,x+1)


def trainRandomBatchIncremental(nn:NeuralNetwork, x_train:np.ndarray, y_train:np.ndarray,
                batchSize:int, learning_rate:float,batchIteration:int=1):
    batchAmount=x_train.shape[1]//batchSize
    n_permutation= np.random.permutation(x_train.shape[1])
    x_train=x_train[:,n_permutation]
    y_train=y_train[:,n_permutation]
    for (x,x1) in binaryIteration(batchAmount):
        batchIndex=x*batchSize
        batchIndex1=x1*batchSize
        x_batch=x_train[:,batchIndex:batchIndex1]
        y_batch=y_train[:,batchIndex:batchIndex1]
        yield nn.train_batch(x_batch,y_batch,learning_rate,batchIteration)


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
    (x_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist('./data/mnist.pkl.gz')
    
    print("Dataset shapes:")
    print(f"X_train: {x_train.shape}, y_train: {y_train.shape}")
    
    # Create neural network: 784 inputs, 128 hidden, 10 outputs
    print("Creating neural network...")
#    nn:NeuralNetwork=load_parameters("./temp.pkl")
    nn:NeuralNetwork= load_parameters("./mnist_model.pkl")
    # Train with mini-batches
    print("Training neural network...")
    for x in trainRandomBatchIncremental(nn,x_train,y_train,32,0.1,30):
        print(f"error:{x}")
    nn.save_parameters("mnist_model.pkl")
    # Evaluate on some test samples
    
