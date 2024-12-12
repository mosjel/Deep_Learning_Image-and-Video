import numpy as np
h=np.empty([2,2])
h[1,1]=1
h[0,0]=100.4444
h=h.round(2)
print(h)