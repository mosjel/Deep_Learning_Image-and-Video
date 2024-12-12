import numpy as np
import pandas as pd

# Sample DataFrame with integer list column
df = pd.DataFrame({'A': [[1, 2, 3], [4, 5, 6],[9,9,9], [7, 8, 9],[2,2,2]]})

# Target list
target_list = [2, 5, 8]

# Convert the integer list column to a NumPy array
a = np.array(df['A'].tolist())
print(a.shape)
# Find the rows in the NumPy array that contain the target list elements
mask = np.any(np.isin(a, target_list), axis=1)

# Get the row indexes where the mask is True
indexes = df.index[(np.any(np.isin(a, target_list), axis=1))].tolist()

print(indexes)