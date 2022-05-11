from copy import deepcopy
import sys
import numpy as np
from scipy.linalg import null_space
from config import logger, rank, comm
from mpi4py import MPI
import time

from federated_xgboost.XGBoostCommon import  PARTY_ID, SplittingInfo
from data_structure.TreeStructure import *
from data_structure.DataBaseStructure import *
from federated_xgboost.XGBoostCommon import compute_splitting_score, XgboostLearningParam
from federated_xgboost.FLTree import FLXGBoostClassifierBase, FLPlainXGBoostTree

class FEDXGBOOST_MSGID:
    SECURE_KERNEL = 200
    SECURE_RESPONSE = 201
    OPTIMAL_SPLITTING_SELECTION = 202
    OPTIMAL_SPLITTING_INFO = 203

class FEDXGBOOST_PARAMETER:
    nMaxResponse = 20

def secure_response(privateX, U):
    # r = np.random.randint(U.shape[1])
    # Z = U[:, np.random.randint(U.shape[1], size=r)]
    # W = np.identity(privateX.shape[0]) - np.matmul(Z, Z.T)
    return privateX
    return np.matmul(W,privateX)



class FedXGBoostClassifier(FLXGBoostClassifierBase):
    def __init__(self, nTree = 3):
        trees = []
        for _ in range(nTree):
            tree = VerticalFedXGBoostTree()
            trees.append(tree)
        super().__init__(trees)


class VerticalFedXGBoostTree(FLPlainXGBoostTree):
    def __init__(self, param: XgboostLearningParam = XgboostLearningParam()):
        super().__init__(param)

    # Child class declares the privacy optimal split finding
    def fed_optimal_split_finding(self, qDataBase: QuantiledDataBase):    
        # Start finding the optimal candidate federatedly
        nprocs = comm.Get_size()
        sInfo = SplittingInfo()
        privateSM = qDataBase.get_merged_splitting_matrix()

        if rank == PARTY_ID.ACTIVE_PARTY:
            startOptimalFinding = time.time()
            
            if privateSM.size: # check it own candidate
                score, maxScore, bestSplitId = compute_splitting_score(privateSM, qDataBase.gradVec, qDataBase.hessVec)
                if(maxScore > 0):
                    sInfo.isValid = True
                    sInfo.bestSplitParty = PARTY_ID.ACTIVE_PARTY
                    sInfo.selectedCandidate = bestSplitId
                    sInfo.bestSplitScore = maxScore

            # Perform the QR Decomposition
            matGH = np.concatenate((qDataBase.gradVec, qDataBase.hessVec), axis=1)

            #print(matGH)
            #q, r = np.linalg.qr(matGH)
            
            Z = null_space(matGH.T)
            indices = np.random.choice(Z.shape[1], FEDXGBOOST_PARAMETER.nMaxResponse, replace=False)
            Z = Z[:, indices]
            
            logger.debug("Performed QR decomposition of [G, H])")
            logger.debug("GradHess matrix: %s", str(matGH.T))
            logger.debug("Nullspace matrix: %s", str(Z))
            for partners in range(2, nprocs):   
                # Send the Secure kernel to the PP
                status = comm.send(Z, dest = partners, tag = FEDXGBOOST_MSGID.SECURE_KERNEL)
                #logger.warning("Sent the splitting matrix to the active party")  
                self.commLogger.log_nTx(Z.size * Z.itemsize, partners)

            for i in range(2, nprocs):
                # Receive the Secure Response from the PP
                stat = MPI.Status()
                rxSR = comm.recv(source=MPI.ANY_SOURCE, tag = FEDXGBOOST_MSGID.SECURE_RESPONSE, status = stat)
                logger.info("Received the secure response from the passive party")
                self.commLogger.log_nRx(rxSR.size * rxSR.itemsize, stat.Get_source())

                # Collect all private splitting info from the partners to find the optimal splitting candidates
                # Find the optimal splitting score iff the splitting matrix is provided
                rxSR = rxSR.T
                if rxSR.size:
                    score, maxScore, bestSplitId = compute_splitting_score(rxSR, qDataBase.gradVec, qDataBase.hessVec)
                    # Select the optimal over all partner parties
                    if (maxScore > sInfo.bestSplitScore):
                        sInfo.bestSplitScore = maxScore
                        sInfo.bestSplitParty = stat.Get_source()
                        sInfo.selectedCandidate = bestSplitId
                        sInfo.isValid = True
                    
       
            # Build Tree from the feature with the optimal index
            for partners in range(2, nprocs):
                data = comm.send(sInfo, dest = partners, tag = FEDXGBOOST_MSGID.OPTIMAL_SPLITTING_SELECTION)

            print("Time optimal finding",time.time() - startOptimalFinding)
        
        elif (rank != 0):           
            # Perform the secure Sharing of the splitting matrix
            privateSM = qDataBase.get_merged_splitting_matrix()
            logger.info("Merged splitting options from all features and obtain the private splitting matrix with shape of {}".format(str(privateSM.shape)))
            logger.debug("Value of the private splitting matrix is \n{}".format(str(privateSM))) 

            # Receive the secured kernel
            secureKernel = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = FEDXGBOOST_MSGID.SECURE_KERNEL)
            logger.info("Received the secure kernel from the active party")

            # Compute the secure response and send to the active party
            secureRep = np.array([], dtype=np.float32)
            if(privateSM.size):
                #compute_time = time.time()
                secureRep = secure_response(privateSM.T, secureKernel)
                #print("Secure Response Time", time.time() - compute_time)        
                logger.info("Sent the secure response to the active party")
            else:
                logger.info("No splitting option feasible. Sent empty splitting matrix the active party")
            status = comm.send(secureRep, dest=PARTY_ID.ACTIVE_PARTY, tag = FEDXGBOOST_MSGID.SECURE_RESPONSE)

            # Receive the optimal splitting information
            sInfo = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = FEDXGBOOST_MSGID.OPTIMAL_SPLITTING_SELECTION)            
            logger.info("Received the Splitting Info from the active party") 
        
            
        # Post processing, final announcement (optimal splitting vector)
        sInfo = self.fed_finalize_optimal_finding(sInfo, qDataBase, privateSM)
        
        return sInfo



   
