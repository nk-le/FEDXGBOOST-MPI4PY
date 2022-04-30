import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from data_structure.DataBaseStructure import *
from data_structure.TreeStructure import *
from fedxgboost.FedXGBoostTree import VerticalFedXGBoostTree
from common.Common import PARTY_ID, logger, rank, clientNum
from algo.LossFunction import LogLoss, LeastSquareLoss



class FedXGBoostClassifier():
    def __init__(self, nTree = 3):
        self.nTree = 3
        self.trees = []
        for _ in range(nTree):
            tree = VerticalFedXGBoostTree()
            self.trees.append(tree)

        self.dataBase = DataBase()
        self.label = []

    def append_data(self, dataTable, fName = None):
        """
        Dimension definition: 
        -   dataTable   nxm: <n> users & <m> features
        -   name        mx1: <m> strings
        """
        self.dataBase = DataBase.data_matrix_to_database(dataTable, fName)
        logger.warning('Appended data feature %s to database of party %d', str(fName), rank)

    def append_label(self, labelVec):
        self.label = np.reshape(labelVec, (len(labelVec), 1))

    def print_info(self):
        featureListStr = '' 
        ret = self.dataBase.log()
        print(ret)

    def boost(self):
        # TODO: the data is passed in the method append data
        orgData = deepcopy(self.dataBase)
        y = self.label
        y_pred = np.zeros(np.shape(self.label))
        
        for i in range(self.nTree):     
            # Perform tree boosting
            dataFit = QuantiledDataBase(self.dataBase)
            self.trees[i].fit_fed(y, y_pred, i, dataFit)

            if i == self.nTree - 1: # The last tree, no need for prediction update.
                continue
            else:
                update_pred = self.trees[i].predict_fed(orgData)
            if rank == PARTY_ID.ACTIVE_PARTY:
                update_pred = np.reshape(update_pred, (self.dataBase.nUsers, 1))
                y_pred += update_pred

    def predict_fed(self, X, fName = None):
        y_pred = None
        data_num = X.shape[0]
        # Make predictions
        testDataBase = DataBase.data_matrix_to_database(X, fName)
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict_fed(testDataBase)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred).reshape(data_num, -1)
            if rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred
        return y_pred
