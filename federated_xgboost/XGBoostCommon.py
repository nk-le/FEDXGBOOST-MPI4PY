
import numpy as np
from algo.LossFunction import LeastSquareLoss, LogLoss


def compute_splitting_score(SM, GVec, HVec, lamb):
    G = sum(GVec)
    H = sum(HVec)
    GRVec = np.matmul(SM, GVec)
    HRVec = np.matmul(SM, HVec)
    GLVec = G - GRVec
    HLVec = H - HRVec
    # logger.info("Received from party {} \n".format(partners) + \
    #     "GR: " + str(sumGRVec.T) + "\n" + "HR: " + str(sumHRVec.T) +\
    #     "\nGL: " + str(sumGLVec.T) + "\n" + "HL: " + str(sumHLVec.T))  

    L = (GLVec*GLVec / (HLVec + lamb)) + (GRVec*GRVec / (HRVec + lamb)) - (G*G / (H + lamb))
    return L.reshape(-1)


class XgboostLearningParam:
    def __init__(self) -> None:
        self.LOSS_FUNC = LogLoss()
        self.LAMBDA = 1
        self.GAMMA = 0.5
        self.EPS = 0.1
        self.N_TREES = 3
        self.MAX_DEPTH = 3

