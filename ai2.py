from typing import Callable, List
import numpy as np
import numpy.random as ran
import gzip
import pickle
#Sigmoid function is =e^x/(1+e^x)
def sigmoid(x:np.ndarray):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(sigmoid_value:np.ndarray):
    return sigmoid_value*(1-sigmoid_value)
def sq_error(output:np.ndarray,expected:np.ndarray)->float:
    diff=output-expected
    return diff.dot(diff)
def sq_error_derivative(output:np.ndarray,expected:np.ndarray)->np.ndarray:
    return 2*(output-expected)
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
        self.weight=[ran.randn(y,x) for x,y in zip(layers_dim,layers_dim[1:])]
        self.bias=[ran.randn(x,1) for x in layers_dim[1:]]
        # I had question,What If I had added bais in the first input,it would have made better support.

        self.neuron=[np.zeros((x,1)) for x in layers_dim]
        self.non_linear_func=non_linear_func
        self.non_linear_func_derivitave=non_linear_func_derivitave
        self.errorFunction=errorFunction
        self.ErrorFunctionDerivative=errorFuntionDerivative
    
    def forward_pass(self,input:np.ndarray=None):
        if input is not None:
            self.neuron[0][...]=input
        for weight,bais,neuron,neuron1 in zip(self.weight,self.bias,self.neuron,self.neuron[1:]):
            neuron1[...]=self.non_linear_func(weight @ neuron + bais)
    def backward_pass(self,expected:np.ndarray,learning_rate:float):
        print("Error before:",self.errorFunction(self.neuron[-1],expected))
        in_derivative=self.ErrorFunctionDerivative(self.neuron[-1],expected)*self.non_linear_func_derivitave(self.neuron[-1])
        for w,b,n in zip(self.weight[::-1],self.bias[::-1],self.neuron[:-1][::-1]):
            b_derivative=in_derivative
            w_derivative=in_derivative @ n.T
            in_derivative=np.dot(in_derivative,w)*self.non_linear_func_derivitave(n)
            w[...]+=-learning_rate*w_derivative
            b[...]+=-learning_rate*b_derivative
        self.forward_pass()
        print("Error now:",self.errorFunction(self.neuron[-1],expected))

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

def main():
    n=NeuralNetwork([784,16,16,10])
    train






        

