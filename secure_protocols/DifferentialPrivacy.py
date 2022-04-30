from locale import normalize
import pydp as dp
import numpy as np
import matplotlib.pyplot as plt

def perturb_laplace(x, eps):
    noise = np.random.laplace(loc = x, scale=eps)
    print(noise)
    return x + noise

def test():
    x = np.random.permutation(100*1).reshape(100,1)
    x = x / np.linalg.norm(x)
    xP = perturb_laplace(x, 1/0.1)
    xP = xP / np.linalg.norm(xP)
    plt.plot(x, ".")
    plt.plot(xP, ".")
    plt.legend(["raw", "pert"])
    plt.show()
test()