
import numpy as np
from algo.LossFunction import LeastSquareLoss, LogLoss

L = lambda G,H, GL, GR, HL, HR, lamb: (GL*GL / (HL + lamb)) + (GR*GR / (HR + lamb)) - (G*G / (H + lamb))

def compute_splitting_score(SM, GVec, HVec, lamb = 0.1):
    G = sum(GVec)
    H = sum(HVec)
    GRVec = np.matmul(SM, GVec)
    HRVec = np.matmul(SM, HVec)
    GLVec = G - GRVec
    HLVec = H - HRVec
    score = L(G,H,GLVec,GRVec,HLVec,HRVec,lamb)
    return score.reshape(-1)

def get_splitting_score(G, H, GL, GR, HL, HR, lamb = 0.1):
    score = L(G,H,GL,GR,HL,HR,lamb)
    return score.reshape(-1)

class XgboostLearningParam:
    def __init__(self) -> None:
        self.LOSS_FUNC = LogLoss()
        self.LAMBDA = 1
        self.GAMMA = 0.5
        self.EPS = 0.1
        self.N_TREES = 3
        self.MAX_DEPTH = 3

