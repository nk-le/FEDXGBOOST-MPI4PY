import math
import argparse
import numpy as np
from typing import Union
import scipy.linalg

def get_nullspace_vector(A, nNSVec = 0):
    """
    Applies the Gram-Schmidt method to A
    and returns Q and R, so Q*R = A.
    """
    m = A.shape[0]
    n = A.shape[1]
    print(m,n)
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    retNS = np.zeros((A.shape[0], nNSVec))
    i = 0
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        Q[:, k] = A[:, k]/R[k, k]
        for j in range(k+1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] = A[:, j] - R[k, j]*Q[:, k]

        if((i < nNSVec)):
            if(k >= (m-n)):
                retNS[:, i] = Q[:, k]
                i += 1
                break

    return retNS

mat = np.random.randint(10, size = (10,3))
retNS = get_nullspace_vector(mat, 3)

print(retNS)


