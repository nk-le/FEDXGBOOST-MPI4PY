import numpy as np
import pandas as pd
from datetime import *
from math import ceil, log
import time

from config import logger
from federated_xgboost.XGBoostCommon import XgboostLearningParam, SplittingInfo

class TreeNodeType:
    ROOT = "Root"
    LEFT = "Left"
    RIGHT = "Right"
    LEAF = "Leaf"


class TreeNode:
    def __init__(self, weight = 0.0, leftBranch=None, rightBranch=None):
        self.weight = weight
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        

    def logNode(self):
        logger.info("Child Node Addresses: L %d| R %d", id(self.leftBranch), id(self.rightBranch))

    def get_string_recursive(self):
        str = ""
        if not self.is_leaf():
            str += "[Addr: {} Child L: {} Child R: {} Weight: {}]".format(id(self), id(self.leftBranch), id(self.rightBranch), self.weight)
            str += "{}".format(self.get_private_info())
            str += " \nChild Info \nLeft Node: {} \nRight Node: {}".format(self.leftBranch.get_string_recursive(), self.rightBranch.get_string_recursive())
        else:
            str += "[TreeLeaf| Addr: {} Weight: {}]".format(id(self), self.weight)
        return str

    def get_private_info(self):
        return

    def is_leaf(self):
        return (self.leftBranch is None) and (self.rightBranch is None)


class FLTreeNode(TreeNode):
    def __init__(self, FID = 0, weight=0, nUsers = 0, leftBranch=None, rightBranch=None, ownerID = -1):
        super().__init__(weight, leftBranch, rightBranch)
        self.FID = FID
        self.owner = ownerID
        self.splittingInfo = SplittingInfo()
        self.nUsers = nUsers
        self.score = None

    def get_private_info(self):
        return "\nOwner ID:{}".format(self.owner)

    def set_splitting_info(self, sInfo: SplittingInfo):
        self.owner = sInfo.bestSplitParty
        self.splittingInfo = sInfo

    def find_child_node(self, id):
        if (self.FID) is id:
            return self
        for child in [self.leftBranch, self.rightBranch]:
            if child is not None:
                ret = child.find_child_node(id)
                if ret:
                    #print("Yay")
                    return ret
        return None

  
    def compute_score(self):
        score = 0
        if self.is_leaf():
            return self.score + XgboostLearningParam.GAMMA
        else:
            for child in [self.leftBranch, self.rightBranch]:
                if child is not None:
                    score += child.compute_score() 
        return score

    @staticmethod
    def compute_leaf_param(gVec, hVec, lamb = XgboostLearningParam.LAMBDA):
        gI = sum(gVec) 
        hI = sum(hVec)
        weight = -1.0 * gI / (hI + lamb)
        score = 1/2 * weight * gI
        return weight, score

