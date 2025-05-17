import numpy as np
from typing import List
class WeightBaseList:
    def __init__(self,neuronDim:List[int],empty:bool=False) -> None:
        self.neuronDim=neuronDim
        totalWeight=sum([n*n1 for n,n1 in zip(neuronDim,neuronDim[1:])])
        totalBase=sum(neuronDim[1:])
        if empty:self.core=np.zeros_like(totalWeight+totalBase)
        else: self.core=np.random.rand(totalWeight+totalBase)-0.5
        self.weightList=[]
        self.biasList=[]
        previous=0
        for n,n1 in zip(neuronDim,neuronDim[1:]):
            size=n*n1
            self.weightList.append(self.core[previous:previous+size].reshape((n1,n)))
            previous+=size
        for n in neuronDim[1:]:
            self.biasList.append(self.core[previous:previous+n].reshape((n,1)))
            previous+=n

