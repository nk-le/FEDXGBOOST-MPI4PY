
import numpy as np
from algo.LossFunction import LeastSquareLoss, LogLoss

L = lambda G,H, GL, GR, HL, HR, lamb, gamma: 1/2 * ((GL*GL / (HL + lamb)) + (GR*GR / (HR + lamb)) - (G*G / (H + lamb))) - gamma


class XgboostLearningParam:
    #def __init__(self) -> None:
    LOSS_FUNC = LogLoss()
    LAMBDA = 1
    GAMMA = 0.5
    EPS = 0.1
    N_TREES = 3
    MAX_DEPTH = 3

def compute_splitting_score(SM, GVec, HVec, lamb = XgboostLearningParam.LAMBDA, gamma = XgboostLearningParam.GAMMA):
    G = sum(GVec)
    H = sum(HVec)
    GRVec = np.matmul(SM, GVec)
    HRVec = np.matmul(SM, HVec)
    GLVec = G - GRVec
    HLVec = H - HRVec
    score = L(G,H,GLVec,GRVec,HLVec,HRVec,lamb, gamma)
    return score.reshape(-1)

def get_splitting_score(G, H, GL, GR, HL, HR, lamb = XgboostLearningParam.LAMBDA, gamma = XgboostLearningParam.GAMMA):
    score = L(G,H,GL,GR,HL,HR,lamb, gamma)
    return score.reshape(-1)


