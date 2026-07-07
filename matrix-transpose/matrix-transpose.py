import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    rows, cols = np.array(A).shape
    transpose = np.empty((cols,rows))
    for i in range(rows):
        for j in range(cols):
            transpose[j,i] = np.array(A)[i,j]
    return transpose
