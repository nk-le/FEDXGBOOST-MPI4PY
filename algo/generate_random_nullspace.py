import random
import numpy as np
from scipy.linalg import null_space
 
np.random.seed(10)

def get_approx_nullspace(X, rRandomSelected = None, nOut = 10):
    """
    X is m x n matrix
    """

    m = X.shape[0]
    n = X.shape[1]
    
    # select r random columns vector in X, r >> min(m,n), don't care the other vectors
    #
    if(rRandomSelected is None):
        r = np.random.randint(low = max(m,n)/2, high= max(m,n))
    else:
        r = rRandomSelected
    selColIndex = random.sample(range(1, n), r) 
    # perform the qr decomposition only on the sparse matrix with lower dimension (2 * r)
    # the nullspace vectors of sparseX is also the nullspace of X
    sparseX = X[:, selColIndex]
    Z = null_space(sparseX)
    nOutAvailable = min(nOut, Z.shape[1])
    retNullspace = np.zeros((n, nOutAvailable))
    retNullspace[selColIndex, :nOutAvailable] = Z[:, :nOutAvailable] 

    # print(selColIndex)
    # print("r", r)
    # print("Z", Z.shape)
    # print("X", sparseX.shape)
    return retNullspace 


def test():
    mat = np.array([range(2,50), range(2,50)])
    ns = get_approx_nullspace(mat,20,15)
    print("NS", ns.shape)
    print("Mat", mat.shape)
    print(np.matmul(mat, ns))





