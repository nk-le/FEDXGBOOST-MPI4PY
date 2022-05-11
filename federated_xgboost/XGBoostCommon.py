
from distutils.log import Log
import numpy as np
from algo.LossFunction import LeastSquareLoss, LogLoss
from common.BasicTypes import Direction
from config import logger, rank

L = lambda G,H, GL, GR, HL, HR, lamb, gamma: 1/2 * ((GL*GL / (HL + lamb)) + (GR*GR / (HR + lamb)) - (G*G / (H + lamb))) - gamma

class PARTY_ID:
    ACTIVE_PARTY = 1

class XgboostLearningParam:
    #def __init__(self) -> None:
    LOSS_FUNC = LogLoss()
    LAMBDA = 1
    GAMMA = 0.5
    N_TREES = 10
    MAX_DEPTH = 5

def compute_splitting_score(SM, GVec, HVec, lamb = XgboostLearningParam.LAMBDA, gamma = XgboostLearningParam.GAMMA):
    G = sum(GVec)
    H = sum(HVec)
    GRVec = np.matmul(SM, GVec)
    HRVec = np.matmul(SM, HVec)
    GLVec = G - GRVec
    HLVec = H - HRVec
    score = L(G,H,GLVec,GRVec,HLVec,HRVec,lamb, gamma)

    bestSplitId = np.argmax(score)
    maxScore = score[bestSplitId]
    return score.reshape(-1), maxScore, bestSplitId

def get_splitting_score(G, H, GL, GR, HL, HR, lamb = XgboostLearningParam.LAMBDA, gamma = XgboostLearningParam.GAMMA):
    score = L(G,H,GL,GR,HL,HR,lamb, gamma)
    return score.reshape(-1)
 

class SplittingInfo:
    def __init__(self) -> None:
        self.bestSplitScore = -np.Infinity
        self.bestSplitParty = None
        self.bestSplittingVector = None
        self.selectedFeatureID = 0
        self.selectedCandidate = 0
        self.isValid = False

        self.featureName = None
        self.splitValue = 0.0

    def log(self):
        logger.debug("Best Splitting Score: L = %.2f, Selected Party %s",\
                self.bestSplitScore, str(self.bestSplitParty))
        logger.debug("%s", self.get_str_split_info())
        logger.debug("The optimal splitting vector: %s| Feature ID: %s| Candidate ID: %s",\
            str(self.bestSplittingVector), str(self.selectedFeatureID), str(self.selectedCandidate))


    def get_str_split_info(self):
        """
        
        """
        retStr = ''
        if(self.bestSplittingVector is not None):
            retStr = "[P: %s, N = %s, " % (str(self.bestSplitParty), str(len(self.bestSplittingVector)))
        else:
            return "Infeasible splitting option. The tree growing should be terminated..."

        
        if(self.featureName is not None): # implies the private splitting info is set by the owner party
            retStr += "F: %s, S: %.4f]" % (str(self.featureName), (self.splitValue))
        else:
            retStr += "F: Unknown, s: Unknown]" 
        return retStr


class FedQueryInfo:
    def __init__(self, userIdList = None) -> None:
        #self.nUsers = len(userIdList)
        self.nUsers = 1
        self.userIdList = userIdList

class FedDirRequestInfo(FedQueryInfo):
    def __init__(self, userIdList) -> None:
        super().__init__(userIdList)
        self.nodeFedId = None

    def log(self):
        logger.debug("Inference Request| NodeFedID %d| nUsers: %d| Users: %s|", self.nodeFedId, self.nUsers, self.userIdList)


class FedDirResponseInfo(FedQueryInfo):
    def __init__(self, userIdList) -> None:
        super().__init__(userIdList)
        self.Direction = [Direction.DEFAULT for i in range(self.nUsers)]


