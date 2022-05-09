from matplotlib import use
import numpy as np
from mpi4py import MPI
import logging
from datetime import date
import time 
import os

np.random.seed(10)
clientNum = 4
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger = logging.getLogger()
day = date.today().strftime("%b-%d-%Y")

curTime = round(time.time())

logName = 'Log/{}/{}/FedXGBoost_{}.log'.format(str(day), str(curTime), str(rank))
os.makedirs(os.path.dirname(logName), exist_ok=True)

file_handler = logging.FileHandler(logName, mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
formatter = logging.Formatter('%(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

class TreeNodeType:
    ROOT = "Root"
    LEFT = "Left"
    RIGHT = "Right"
    LEAF = "Leaf"

class TreeEntity:
    def __init__(self) -> None:
        pass


class TreeNode(TreeEntity):
    def __init__(self) -> None:
        self.weight = 0
        self.leftBranch = None
        self.rightBranch = None


class PARTY_ID:
    ACTIVE_PARTY = 1


class MSG_ID:
    MASKED_GH = 99
    RAW_SPLITTING_MATRIX = 98
    OPTIMAL_SPLITTING_INFO = 97

    REQUEST_DIRECTION = 96
    RESPONSE_DIRECTION = 95
    OPTIMAL_SPLITTING_SELECTION = 94
    INIT_INFERENCE_SIG = 89
    ABORT_INFERENCE_SIG = 90

class SplittingInfo:
    def __init__(self) -> None:
        self.bestSplitScore = -np.Infinity
        self.bestSplitParty = None
        self.bestSplittingVector = None
        self.selectedFeatureID = 0
        self.selectedCandidate = 0

        self.featureName = None
        self.splitValue = 0.0

    def log(self):
        logger.info("Best Splitting Score: L = %.2f, Selected Party %s",\
                self.bestSplitScore, str(self.bestSplitParty))
        logger.info("%s", self.get_str_split_info())
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

class Direction:
    DEFAULT = None
    LEFT = 0
    RIGHT = 1

class FedDirResponseInfo(FedQueryInfo):
    def __init__(self, userIdList) -> None:
        super().__init__(userIdList)
        self.Direction = [Direction.DEFAULT for i in range(self.nUsers)]





# class PARTY_ID:
#     ACTIVE_PARTY = 1


# class MSG_ID:
#     MASKED_GH = 99


# np.random.seed(10)
# clientNum = 4
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()


# logger = logging.getLogger()
# logName = 'Log/FedXGBoost_%d.log' % rank
# file_handler = logging.FileHandler(logName, mode='w')
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# logger.setLevel(logging.DEBUG)

# logger.warning("Hello World")


