from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import math
from numpy.linalg import norm
s=[[1,-2,-3],[1,-2,-3]]
s=np.array(s)
print(s.shape)
f=[1,-2,-3]
f=np.array(f)
print(f.shape)

d=1-(np.dot(s,f)/(norm(s,axis=1)*norm(f)))
print(d)