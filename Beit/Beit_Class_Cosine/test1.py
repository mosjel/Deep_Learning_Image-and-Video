import numpy as np
# Create a numpy array from a list of numbers
arr = np.array([[11, 12, 13],[ 14, 15, 16], [17, 15, 11], [12, 14, 15], [16, 17,18]])

result = np.where(arr == 15)
print('Tuple of arrays returned : ', result)
print("Elements with value 15 exists at following indices", result[0], sep='\n')