import numpy as np
a=[np.array([1,2,3,4]),np.array([2,23,4,5])]
b=a.copy()
a[0][1]=5
print(b)
