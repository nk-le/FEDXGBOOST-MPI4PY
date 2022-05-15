import matlab.engine
import time
import numpy as np

mlEngine = matlab.engine.start_matlab()
s = mlEngine.genpath('matlab_algo/receursive-nystrom')
mlEngine.addpath(s, nargout = 0)
s = mlEngine.genpath('matlab_algo/')
mlEngine.addpath(s, nargout = 0)



def test_recursive_nystrom(X):
    # generate random test matrix
    X = matlab.double(X)
    [C,W] = mlEngine.nystrom_wrapper(X, nargout=2)
    print("Hi")



myVec = np.random.random((5,2))
myMat = np.matmul(myVec, myVec.transpose())


test_recursive_nystrom(myMat)


    # // K = matlab.double(X)
    # // #define function for computing kernel dot product
    # // gamma = 20
    # // kFunc = lambda X,rowInd,colInd: mlEngine.gaussianKernel(K,rowInd,colInd,gamma)

    # // #compute factors of Nyström approximation
    # // [C,W] = mlEngine.recursiveNystrom(X,500,kFunc)
