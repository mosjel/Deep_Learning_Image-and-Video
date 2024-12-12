import numpy as np
from keras import models,layers
s=[[[[1,2,3],[4,5,6],[1,1,1]],[[7,8,9],[10,11,12],[2,2,2]],[[1,1,1],[2,2,2],[3,3,3]]]]
s=np.array(s,dtype=np.float32)
g=layers.Conv2D(5,(2,2))
hamed=g(s)
g1=layers.Flatten()
hamed1=g1(hamed)


print(hamed.shape)
print(hamed.shape)
print(hamed1.shape)

