from typing import Callable, List
import numpy as np
import numpy.random as ran
import numpy.typing as npt


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

        self.neuron=[np.empty_like(x) for x in layers_dim]
        self.non_linear_func=non_linear_func
        self.non_linear_func_derivitave=non_linear_func_derivitave
        self.errorFunction=errorFunction
        self.ErrorFunctionDerivative=errorFuntionDerivative
    
    def forward_pass(self,input:npt.NDArray=None):
        if input is not None:
            self.neuron[0][...]=input
        for weight,bais,neuron,neuron1 in zip(self.weight,self.bias,self.neuron,self.neuron[1:]):
            neuron1[...]=weight @ neuron + bais
            neuron1[...]=self.non_linear_func(neuron1)
    def backward_pass(self,expected:npt.NDArray,learning_rate:float):
        print("Error before:",self.errorFunction(self.neuron[-1],expected))
        in_derivative=self.ErrorFunctionDerivative(self.neuron[-1],expected)
        for w,b,i in zip(self.weight,self.bias,self.neuron)[::-1]:
            b_derivative=in_derivative
            w_derivative=in_derivative @ i.T
            in_derivative=np.dot(in_derivative,w)
            w[...]+=-learning_rate*w_derivative
            b[...]+=-learning_rate*b_derivative
        self.forward_pass()
        print("Error now:",self.errorFunction(self.neuron[-1],expected))







        

