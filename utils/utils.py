import numpy as np

indices = np.array([
    [23,  5, 150, 34, 89],  # Indices of the 5 nearest neighbors for query 1
    [10, 14, 200, 8, 76],   # Indices of the 5 nearest neighbors for query 2
    [7, 21, 53, 19, 81]     # Indices of the 5 nearest neighbors for query 3
])

print(indices[0])