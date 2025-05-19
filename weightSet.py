
import numpy as np
from typing import List, Optional
EMPTY=0
ZEROS=1
RAND=2

class WeightBaseList:
    def __init__(self, neuron_dim: List[int], type:int=RAND) -> None:
        self.neuron_dim = neuron_dim

        # Calculate total elements needed
        self.total_weights = sum(n_in * n_out for n_in, n_out in zip(neuron_dim, neuron_dim[1:]))
        self.total_biases = sum(neuron_dim[1:])
        self.total_size = self.total_weights + self.total_biases

        # Allocate the flat core array
        if type==ZEROS:
            self.core = np.zeros(self.total_size, dtype=np.float32)
        elif type==RAND:
            self.core = np.random.rand(self.total_size).astype(np.float32) - 0.5
        else:
            self.core=np.empty(self.total_size,dtype=np.float32)

        # Build structured views
        self.weight_list, self.bias_list = self._build_views()

    def _build_views(self):
        weight_list = []
        bias_list = []
        idx = 0

        # Layer weights
        for n_in, n_out in zip(self.neuron_dim, self.neuron_dim[1:]):
            size = n_in * n_out
            weights = self.core[idx:idx+size].reshape(n_out, n_in)
            weight_list.append(weights)
            idx += size

        # Biases
        for n_out in self.neuron_dim[1:]:
            biases = self.core[idx:idx+n_out].reshape(n_out, 1)
            bias_list.append(biases)
            idx += n_out

        return weight_list, bias_list

    def get_weights(self, layer: int) -> np.ndarray:
        return self.weight_list[layer]

    def get_biases(self, layer: int) -> np.ndarray:
        return self.bias_list[layer]

    def copy(self) -> "WeightBaseList":
        # Returns a deep copy
        new_obj = WeightBaseList(self.neuron_dim, type=EMPTY)
        new_obj.core[:] = self.core.copy()
        return new_obj

    def __repr__(self) -> str:
        return f"WeightBaseList(layers={self.neuron_dim}, shape={self.core.shape})"
