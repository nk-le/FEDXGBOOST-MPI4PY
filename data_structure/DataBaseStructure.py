from tokenize import String
from matplotlib.pyplot import sca
import numpy as np
import pandas as pd
from scipy import rand

from common.BasicTypes import Direction
from federated_xgboost.XGBoostCommon import SplittingInfo
from config import rank, logger

class QuantileParam:
    epsilon = 0.005
    thres_balance = 0.1

class FeatureData:
    def __init__(self, name, dataVector) -> None:
        self.name = name
        self.data = np.array(dataVector, dtype=np.float32)

class QuantiledFeature(FeatureData):
  
    def __init__(self, name, dataVector) -> None:
        super().__init__(name, dataVector)
        self.splittingMatrix, self.splittingCandidates = QuantiledFeature.quantile(self.data, QuantileParam)     
        
    def quantile(fData: FeatureData, param: QuantileParam):
        splittingMatrix = []
        splittingCandidates = []

        # Credits to HikariX ... copy for convenience refactoring ...
        split_list = []
        data = np.copy(fData.data)
        idx = np.argsort(data)
        data = data[idx]
        value_list = sorted(list(set(list(data))))  # Record all the different value
        hess = np.ones_like(data)
        data = np.concatenate((data, hess), axis=0)
        sum_hess = np.sum(hess)
        last = value_list[0]
        i = 1
        if len(value_list) == 1: # For those who has only one value, do such process.
            last_cursor = last
        else:
            last_cursor = value_list[1]
        split_list.append((-np.inf, value_list[0]))
        while i < len(value_list):
            cursor = value_list[i]
            small_hess = np.sum(data[data <= last]) / sum_hess
            big_hess = np.sum(data[data <= cursor]) / sum_hess
            if np.abs(big_hess - small_hess) < param.epsilon:
                last_cursor = cursor
            else:
                judge = value_list.index(cursor) - value_list.index(last)
                if judge == 1: # Although it didn't satisfy the criterion, it has no more split, so we must add it.
                    split_list.append((last, cursor))
                    last = cursor
                else: # Move forward and record the last.
                    split_list.append((last, last_cursor))
                    last = last_cursor
                    last_cursor = cursor
            i += 1
        if split_list[-1][1] != value_list[-1]:
            split_list.append((split_list[-1][1], value_list[-1]))  # Add the top value into split_list.
        split_list = np.array(split_list)


        # Khanh goes on from here, I need the splittingCandidates
        splittingCandidates = [split_list[i][1] for i in range(len(split_list))]
        splittingMatrix, splittingCandidates = QuantiledFeature.generate_splitting_matrix(fData.data, splittingCandidates)

        # import matplotlib.pyplot as plt 
        # print(split_list)
        # plt.hist(fData.data)
        # plt.hist(sm[6])
        # plt.show()

        return splittingMatrix, splittingCandidates


    def get_splitting_info(self):
        return self.splittingMatrix, self.splittingCandidates

    def generate_splitting_matrix(dataVector, splittingCandidates):
        """
        assign 0 if the data value is smaller than the splitting candidates (left node)
        assign 1 if the data value is bigger than the splitting candidates (left node)
        """
        outSM = []
        outSC = []
        for i in range(len(splittingCandidates)):
            v =  1. * (dataVector > splittingCandidates[i])

            # Returns only the splitting option with balance between the amount of users in both nodes
            nL = np.count_nonzero(v == 0.0)
            nR = np.count_nonzero(v == 1.0)
            isValid = (((nL/len(v)) > QuantileParam.thres_balance) and ((nR/len(v)) > QuantileParam.thres_balance))
            if(isValid):
                outSM.append(v)
                outSC.append(splittingCandidates[i])

        return np.array(outSM), np.array(outSC)


class DataBase:
    def __init__(self) -> None:
        self.featureDict = {}
        self.nUsers = 0


    def append_feature(self, featureData: FeatureData):
        #self.featureData.append(featureData)
        self.featureDict[featureData.name] = featureData.data
        self.nUsers = len(featureData.data)

    def remove_feature(self, key):
        """
        Users must ensure that the key exists
        """
        del self.featureDict[key]

    def log(self):
        fNameList = ''
        for fName, fData in self.featureDict.items():
            fNameList += fName + '; '
        #print("Existing Feature: ", fNameList)
        #print(self.get_data_matrix())
        logger.info("Database Info: %s", fNameList)
        return fNameList

    def get_feature_name(self):
        ret = []
        for fName, fData in self.featureDict.items():
            ret.append(fName)
        return ret

    def get_data_matrix(self, nameFeature = None):
        X = pd.DataFrame(self.featureDict).values
        return X

    def get_direction(self, sInfo: SplittingInfo, userList: list):
        ret = [Direction.DEFAULT for i in range(self.nUsers)]
        leftId = self.featureDict[sInfo.featureName].data[userList] <= sInfo.splitValue
        rightId = self.featureDict[sInfo.featureName].data[userList] > sInfo.splitValue 
        
        ret[leftId] = Direction.LEFT
        ret[rightId] = Direction.RIGHT
        return ret

    @staticmethod
    def data_matrix_to_database(dataTable: np.ndarray, featureName = None):
        nFeatures = len(dataTable[0])
        if(featureName is None):
            featureName = ["Rank_{}_Feature_".format(rank) + str(i) for i in range(nFeatures)]
        
        assert (len(featureName) is nFeatures) # The total amount of columns must match the assigned name 
        
        dataBase = DataBase()
        for i in range(len(featureName)):
            dataBase.append_feature(FeatureData(featureName[i], dataTable[:,i]))
        
        return dataBase

class QuantiledDataBase(DataBase):
    def __init__(self, dataBase:DataBase = None) -> None:
        super().__init__()
        self.nUsers = dataBase.nUsers
        # Perform the quantiled for all the feature (copy, don't change the orgiginal data)
        if dataBase is not None:
            for feature, value in dataBase.featureDict.items():
                self.featureDict[feature] = QuantiledFeature(feature, value)

        self.gradVec = []
        self.hessVec = []

    def get_info_string(self):
        Str = "nUsers: %d nFeature: %d \n" % (self.nUsers, len(self.featureDict.items()))
        for key, feature in self.featureDict.items():
            sm, sc = self.featureDict[key].get_splitting_info()
            Str += "{} splitting candidates of feature {}: ".format(str(len(sc)), key) + " [{}] \n".format(' '.join(map(str, sc)))
        return Str

    def get_merged_splitting_matrix(self):
        retMergedSM = np.array([])
        for key, feature in self.featureDict.items():
            fSM, sc = self.featureDict[key].get_splitting_info()
            if not retMergedSM.size:
                retMergedSM = fSM
            # Append to the total splitting matrix if the quantiled is feasible
            else:
                if fSM.size:
                    retMergedSM = np.concatenate((retMergedSM,fSM))  
        return retMergedSM

    def find_fId_and_scId(self, bestSplitVector):
        """
        Find the optimal splitting feature and threshold corressponding to the optimal splitting vector
        """
        for key, feature in self.featureDict.items():
            fSM, scArr = self.featureDict[key].get_splitting_info()
            for v, s in zip(fSM, scArr):
                if(np.allclose(v, bestSplitVector)):
                    return key, s
                
        logger.error("No matched splitting candidate.")
        logger.error("Optimal splitting vector: %s", str(bestSplitVector))
        logger.error("My splitting matrix: %s", str(self.get_merged_splitting_matrix()))
        assert(False)


    def partition(self, splittingVector):
        """
        Partition the database to two left and right databases according to the spliitng vector
        The returned QuantiledDatabase  perform the quantile (proposal of the splitting matrices within its constructor)
        """
        # assert Numel of splitting vector and the amount of users
        retL = DataBase()
        retR = DataBase()

        for feature, value in self.featureDict.items():
            leftDictData = self.featureDict[feature].data[splittingVector == 0]
            rightDictData = self.featureDict[feature].data[splittingVector == 1]
            
            retL.append_feature(FeatureData(feature, leftDictData))
            retR.append_feature(FeatureData(feature, rightDictData))
        
        retL = QuantiledDataBase(retL)
        retL.appendGradientsHessian(self.gradVec[splittingVector == 0], self.hessVec[splittingVector == 0])

        retR = QuantiledDataBase(retR)
        retR.appendGradientsHessian(self.gradVec[splittingVector == 1], self.hessVec[splittingVector == 1])

        return retL, retR

    def appendGradientsHessian(self, g, h):
        self.gradVec = np.array(g).reshape(-1,1)
        self.hessVec = np.array(h).reshape(-1,1)








def testQuantile():
    vec = rand(100)
    handle = QuantiledFeature("RandomData", vec)
    print("Amount of Splitting Cadndidates: {}" .format(len(handle.splittingCandidates)))

# np.random.seed(66)
# testQuantile()

# data = [[0,1,2,3], [1,2,3,4]]
# data = np.reshape(data, (4, 2))
# print(len(data), len(data[0]))
# print(data[:,0])