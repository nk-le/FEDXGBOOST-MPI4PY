from math import ceil
import random
import numpy as np
from scipy.linalg import null_space
 
np.random.seed(10)


def get_approx_nullspace_secure(X, nOut = 10, nBatch = 2):
    """
    X is m x n matrix
    """

    m = X.shape[0]
    n = X.shape[1]
    
    
    orgList = list(range(m))
    excludeList = []
    while len(excludeList)!= m:
        # select r random columns vector in X, r >> min(m,n), don't care the other vectors    
        # Use random choice and exclude the selected column so that the generated orthogonal matrix does not have zero rows
        tobeSelectedList = list(set(orgList) - set(excludeList))
        selColIndex = list(np.random.choice(tobeSelectedList, size = nBatch))
        excludeList.append(selColIndex) # Add to the ignore list to ensure at the end, no zero rows exist

        
        selColIndex = selColIndex.append(np.random.choice(tobeSelectedList, size = nBatch)) # Actually some overlapping rows is desired. TODO: more investigation on this steps         
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



def get_approx_nullspace(X, rRandomSelected = None, ratio = 10):
    """
    X is m x n matrix
    """

    m = X.shape[0]
    n = X.shape[1]
    
    orgList = range(n)
    nOut = int(ratio * n)
    retNullspace = np.zeros((n, nOut))
    for i in range(nOut):
        # select r random columns vector in X, r >> min(m,n), don't care the other vectors
        selColIndex = random.sample(range(n), ceil((1/ratio))) 
        # perform the qr decomposition only on the sparse matrix with lower dimension (2 * r)
        # the nullspace vectors of sparseX is also the nullspace of X
        sparseX = X[:, selColIndex]
        Z = null_space(sparseX)
        retNullspace[selColIndex, i] = Z[:, 0] 

    # print(selColIndex)
    # print("r", r)
    # print("Z", Z.shape)
    # print("X", sparseX.shape)
    return retNullspace 



def test():
    mat = np.array([range(2,15), range(2,15)])
    ns = get_approx_nullspace(mat,20,3)
    print("NS", ns.shape)
    print("Mat", mat.shape)
    print(np.matmul(mat, ns))

#test()




