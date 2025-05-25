
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
def sigmoid(x:np.ndarray):
    x=np.clip(x,-500,500)
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

def gradientDecent(learningRate:float=1):
              return lambda x:learningRate*x
def maxClip(maxLength:float=1):
    def func(vec:np.ndarray):
     length=np.linalg.norm(vec)
     if length>maxLength:return vec*(maxLength/length)
     return vec
    return func

def momentum(b:float=0.8):
              prevoius_vec=0
              def func(vec:np.ndarray):
                nonlocal prevoius_vec
                ans=b*prevoius_vec+(1-b)*vec
                prevoius_vec=ans
                return ans
              return func

class NeuralNetwork:
    def __init__(self,layers_dim:List[int],
                 non_linear_func:ArrayFunction=sigmoid,
                 non_linear_func_derivitave:ArrayFunction=sigmoid_derivative,
                 errorFunction:ErrorFunction=sq_error,
                 errorFuntionDerivative:ErrorFunctionDerivative=sq_error_derivative,
                 learningAlgorithm:Callable[[np.ndarray],np.ndarray]=lambda x:x):
        """
                Backward pass take the negative direction of the gradient,
              and return how the gradient should be updated.
        """
        self.layers_dim=layers_dim
        self.freeVariable=WeightBaseList(self.layers_dim)
        # Initialize weights and biases as before
        self.weights=self.freeVariable.weight_list
        self.biases=self.freeVariable.bias_list
        # Neuron aren't initialized as array because of batches,As their are veraity of batches
        self.neuron:List[ndarray]=[None for x in layers_dim]
        self.non_linear_func=non_linear_func
        self.non_linear_func_derivitave=non_linear_func_derivitave
        self.errorFunction=errorFunction
        self.errorFunctionDerivative=errorFuntionDerivative
        self.learningAlgorithm=learningAlgorithm
    
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
    def backward_pass(self,expectedOutput:np.ndarray):
       gradient=self.gradient(expectedOutput)
       self.freeVariable.core+=self.learningAlgorithm(-gradient.core)
    # Updated to handle batch inputs
    def gradient(self, expected:np.ndarray):
        """
        Perform backward pass with batch support
        
        Args:
            expected: shape (output_dim, batch_size)
            learning_rate: learning rate for gradient descent
        """
        batch_size = expected.shape[1]
        derivative=self.freeVariable.copy()
        # Initial error derivative for output layer (now handles batches)
        in_derivative = self.errorFunctionDerivative(self.neuron[-1], expected) * \
                        self.non_linear_func_derivitave(self.neuron[-1])
        # in_derivative is now (neuron number, batch size)
        
        # Backpropagate through layers
        for w, dw, db, i in zip(self.weights[::-1],derivative.weight_list[::-1]
                           ,derivative.bias_list[::-1], self.neuron[:-1][::-1]):
            # Batch version of derivatives
            db[...] = np.sum(in_derivative, axis=1,keepdims=True)/batch_size  # Sum across batch dimension
            """
                [[1,2,3],
                 [2,3,4],
                 [3,4,5],],This is the input derivative and having mean in axis=1 means it average in x axis [[2],
                                                                                                              [3],
                                                                                                              [4],]

            """
            dw[...] = (in_derivative @ i.T)/batch_size # Matrix multiplication handles all batch samples

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
            in_derivative = (w.T @ in_derivative) * self.non_linear_func_derivitave(i)
            # In_derivative^T= (in_derivative)^T .w
            #As (AB)^T=B^T.A^T So in_derivative=(inderivativeT @ w)^T
            # Update weights and biases (single update using batch average)
        return derivative
    def costCheck(self,expected:ndarray)->float:
        return self.errorFunction(self.neuron[-1],expected)
    # New method for batch training
    def train_batch_optimized(self, X_batch:np.ndarray, y_batch:np.ndarray
                   ,batch_iteration:int=1,max_batch_size:int=512):
        """
        Train on a single batch
        
        Args:
            X_batch: Input data with shape (input_dim, batch_size)
            y_batch: Expected outputs with shape (output_dim, batch_size)
            learning_rate: Learning rate for gradient descent
            batch_iteration: times a batch is trained
            max_batch_size: If the batch size exceed this,A small part of the batch will be taken
        """
        # Forward pass with batch
        error=0
        batch_size=y_batch.shape[1]
        need_reduced=batch_size>max_batch_size
        reduced_batch_size=int(max_batch_size / 1.3)
        for _ in range(batch_iteration):
            x_m_batch=X_batch
            y_m_batch=y_batch
            if need_reduced:
                index=np.random.randint(0,batch_size,reduced_batch_size)
                x_m_batch=X_batch[:,index]
                y_m_batch=y_batch[:,index]
            self.forward_pass(x_m_batch)
            error+=self.costCheck(y_m_batch)
            self.backward_pass(y_m_batch)
        return error/batch_iteration
    def train_batch(self, X_batch:np.ndarray, y_batch:np.ndarray
                   ,batch_iteration:int=1):
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
            error+=self.costCheck(y_batch)
            self.backward_pass(y_batch)
        return error/batch_iteration

    def predicted_value(self):
        return self.neuron[-1].argmax(axis=0)
    def error_check(self,y_value:np.ndarray):
        return np.mean(self.predicted_value()==y_value.argmax(axis=0))
    # New method for full dataset training with mini-batches
    def save_parameters(self, filename: str):
        np.save(filename,self.freeVariable.core)
    def load_parameters(self,filename:str):
        self.freeVariable.core[...]=np.load(filename)

def gradientCheck():
    epsilon = 1e-4 #The error got higher ,The less the step size,
    nn = NeuralNetwork([30, 10, 10, 20])
    x_input = np.random.randn(30, 10)
    y_input = np.random.randn(20, 10)

    # Compute the cost with original parameters
    nn.forward_pass(x_input)
    cost = nn.costCheck(y_input)

    manualGrad = np.empty_like(nn.freeVariable.core)

    for i in range(len(nn.freeVariable.core)):
        original_value = nn.freeVariable.core[i]

        # Perturb
        nn.freeVariable.core[i] = original_value + epsilon

        # Forward pass and cost with perturbed weight
        nn.forward_pass(x_input)
        cost_new = nn.costCheck(y_input)

        # Restore
        nn.freeVariable.core[i] = original_value

        # Compute finite difference gradient
        manualGrad[i] = (cost_new - cost) / epsilon

    # Compute analytical gradient
    realGrad = nn.gradient(y_input)

    # Difference
    diff = manualGrad - realGrad.core
    print("Max absolute difference:", np.max(np.abs(diff)))
    print("Relative error:", np.linalg.norm(diff) / (np.linalg.norm(manualGrad) + np.linalg.norm(realGrad.core)))
    if np.max(np.abs(diff)) < 1e-4:
        print(f"{GREEN}Gradient check passed!{RESET}")
    else:
        print(f"{RED}Gradient check failed!{RESET}")
        
        # Show largest discrepancies
        worst_indices = np.argsort(np.abs(diff))[-5:]
        print("\nParameters with largest discrepancies:")
        for idx in reversed(worst_indices):
            print(f"Parameter {idx}: Numerical: {manualGrad[idx]}, Analytical: {realGrad.core[idx]}, Diff: {diff[idx]}")
 

def trainBatchall (nn:NeuralNetwork, x_train:np.ndarray, y_train:np.ndarray, 
                   batch_size:int, batchIteration:int=1):
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
            chunkIndex=chunkNo*batch_size
            chunkIndex1=chunkIndex+batch_size
            trainError+=nn.train_batch(X_batch=x_train[:,chunkIndex:chunkIndex1],
                           y_batch=y_train[:,chunkIndex:chunkIndex1],
                           batch_iteration=batchIteration,)
        trainError/=chunkAmount
        return trainError
def trainRandomBatchAll(nn:NeuralNetwork, x_train:np.ndarray, y_train:np.ndarray,
                        batch_size:int, batchIteration:int=1):
    n_data= x_train.shape[1]
    n_permutation= np.random.permutation(n_data)
    return trainBatchall(nn,x_train[:,n_permutation],y_train[:,n_permutation],batch_size,batchIteration)

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
                batchSize:int, batchIteration:int=1):
    batchAmount=x_train.shape[1]//batchSize
    n_permutation= np.random.permutation(x_train.shape[1])
    x_train=x_train[:,n_permutation]
    y_train=y_train[:,n_permutation]
    for (x,x1) in binaryIteration(batchAmount):
        batchIndex=x*batchSize
        batchIndex1=x1*batchSize
        x_batch=x_train[:,batchIndex:batchIndex1]
        y_batch=y_train[:,batchIndex:batchIndex1]
        yield nn.train_batch_optimized(x_batch,y_batch,batchIteration)


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
    nn:NeuralNetwork= NeuralNetwork([784,30,10],learningAlgorithm=momentum())
    nn.load_parameters("./mnist_model.npy")
    # Train with mini-batches
    print("Training neural network...")
#    for x in trainRandomBatchIncremental(nn,x_train,y_train,32,1,30):
#        print(f"error:{x}")
    for x in range(320):
        error=trainRandomBatchAll(nn,x_train,y_train,10);
        print(f"Epoch {x}:Error :{error}")
        nn.forward_pass(X_test)
        print(f"Test Accuracy:{nn.error_check(y_test)}")
    nn.save_parameters("mnist_model.npy")
    # Evaluate on some test samples
    
