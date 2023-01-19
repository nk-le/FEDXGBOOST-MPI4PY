from copy import deepcopy
import sys
from warnings import catch_warnings
import numpy as np
from scipy.linalg import null_space
from algo.generate_random_nullspace import get_approx_nullspace
from config import logger, rank, comm
from mpi4py import MPI
import time

from federated_xgboost.XGBoostCommon import  PARTY_ID, SplittingInfo
from data_structure.TreeStructure import *
from data_structure.DataBaseStructure import *
from federated_xgboost.XGBoostCommon import compute_splitting_score, XgboostLearningParam
from federated_xgboost.FLTree import FLXGBoostClassifierBase, FLPlainXGBoostTree

from config import CONFIG, SIM_PARAM

#import matlab.engine
#from sklearn.preprocessing import normalize

def large_matrix_to_string(mat):
    retStr = ""
    for vec in mat:
        retStr += str(vec) + "\n"    
    return retStr

class FEDXGBOOST_MSGID:
    SECURE_KERNEL = 200
    SECURE_RESPONSE = 201
    OPTIMAL_SPLITTING_SELECTION = 202
    OPTIMAL_SPLITTING_INFO = 203

class FEDXGBOOST_PARAMETER:
    nMaxSecureKernel = 1000 #min(int(SIM_PARAM.N_SAMPLE * 0.11), 3000)
    #nMaxResponse = 30
    nMinSelectedResponse = nMaxSecureKernel
    r = 0.03

    def log():
        logger.warning("FedXGBoostParam, r : %f", FEDXGBOOST_PARAMETER.r)


def secure_response(privateX, U):
    return secure_response_fast(privateX, U)

def plain_response(privateX, U):
    #print("testing", privateX.shape)
    return privateX


def secure_response_plain(privateX, U):
    #r = np.random.randint(U.shape[1])
    #r = min(FEDXGBOOST_PARAMETER.nMinSelectedResponse, U.shape[1])
    r = U.shape[1]
    Z = U[:, np.random.randint(U.shape[1], size=r)]
    W = np.identity(privateX.shape[0]) - np.matmul(Z, Z.T)
    ret = np.matmul(W,privateX)
    return ret
    
def secure_response_fast(privateX, U):
    #r = min(FEDXGBOOST_PARAMETER.nMinSelectedResponse, U.shape[1])
    r = U.shape[1]
    Z = U[:, np.random.randint(U.shape[1], size=r)]
    N = np.random.normal(loc= 1000, scale = U.shape[1]^2, size = (r,privateX.shape[1]))
    #print(Z.shape, N.shape, privateX.shape, U.shape)
    return privateX - np.matmul(Z,N)

# def secure_response_lra_rn(privateX, U):
#     r = np.random.randint(U.shape[1])
#     Z = U[:, np.random.randint(U.shape[1], size=r)]
#     Z = normalize(Z, axis=1, norm='l1')
#     X = matlab.double(Z)
#     from config import mlEngine
#     [C,W,L] = mlEngine.nystrom_wrapper(X, nargout=3)
#     C = np.asarray(C)
#     W = np.asarray(W)
#     L = np.asarray(L)
#     try:
#         #L = np.linalg.cholesky(W)
#         CT_M = np.matmul(C.T, privateX)
#         LT_CT_M = np.matmul(L.T, CT_M)
#         W_CT_M = np.matmul(L, LT_CT_M)
#         KM = np.matmul(C, W_CT_M)
#     except Exception as e:
#         print(e)
#         lamb = np.linalg.eig(W)
#         #print(lamb)

#     return privateX - KM
#     return secure_response_plain(privateX, U)


class FedXGBoostClassifier(FLXGBoostClassifierBase):
    def __init__(self, nTree = 3):
        trees = []
        for i in range(nTree):
            tree = VerticalFedXGBoostTree(i)
            trees.append(tree)
        super().__init__(trees)

        FEDXGBOOST_PARAMETER.log()



class VerticalFedXGBoostTree(FLPlainXGBoostTree):
    def __init__(self, id, param: XgboostLearningParam = XgboostLearningParam()):
        super().__init__(id, param)

    # Child class declares the privacy optimal split finding
    def fed_optimal_split_finding(self, qDataBase: QuantiledDataBase):    
        # Start finding the optimal candidate federatedly
        nprocs = comm.Get_size()
        sInfo = SplittingInfo()
        privateSM = qDataBase.get_merged_splitting_matrix()

        if rank == PARTY_ID.ACTIVE_PARTY:
            #startOptimalFinding = time.time()
            
            if privateSM.size: # check it own candidate
                score, maxScore, bestSplitId = compute_splitting_score(privateSM, qDataBase.gradVec, qDataBase.hessVec)
                if(maxScore > 0):
                    sInfo.isValid = True
                    sInfo.bestSplitParty = PARTY_ID.ACTIVE_PARTY
                    sInfo.selectedCandidate = bestSplitId
                    sInfo.bestSplitScore = maxScore

            # Perform the QR Decomposition
            matGH = np.concatenate((qDataBase.gradVec, qDataBase.hessVec), axis=1)
            # for i in range(10):
            #     tmp =np.random.randint(10, size = (qDataBase.nUsers,1))
            #     matGH = np.concatenate((matGH, tmp), axis=1)


            # ATTENTION: using nullspace directly is not scalable. We use the get_approx_nullspace() instead
            # Z = null_space(matGH.T)
            # # TODO: bring the parameters FEDXGBOOST_PARAMETER.nMaxResponse as parameters outside
            # indices = np.random.choice(Z.shape[1], min(FEDXGBOOST_PARAMETER.nMaxResponse, int((qDataBase.nUsers)/2)), replace=False)
            # Z = Z[:, indices]
            # #print(Z)

            # Select set of coulmn vectors to generate the secure kernel.
            U = get_approx_nullspace(matGH.T, ratio = FEDXGBOOST_PARAMETER.r)

            # Log the nullspace to evaluate the privacy concern
            logger.debug("GradHess matrix: %s", str(matGH.T))
            logger.debug("Transpose of the Generated Nullspace vectors:")
            logger.debug("\n%s", large_matrix_to_string(matGH.T))

            nTx = 0
            for partners in range(2, nprocs):   
                # Send the Secure kernel to the PP
                self.commLogger.log_nTx(U.size * U.itemsize, partners, self.treeID)
                status = comm.send(U, dest = partners, tag = FEDXGBOOST_MSGID.SECURE_KERNEL)
                #logger.warning("Sent the splitting matrix to the active party")         

            for i in range(2, nprocs):
                # Receive the Secure Response from the PP
                stat = MPI.Status()
                rxSR = comm.recv(source=MPI.ANY_SOURCE, tag = FEDXGBOOST_MSGID.SECURE_RESPONSE, status = stat)
                logger.info("Received the secure response from the passive party")
                self.commLogger.log_nRx(rxSR.size * rxSR.itemsize, stat.Get_source(), self.treeID)

                # Collect all private splitting info from the partners to find the optimal splitting candidates
                # Find the optimal splitting score iff the splitting matrix is provided
                if rxSR.size:
                    # Log the SecureResponse rxSR to analyze the privacy level of the secure response
                    logger.debug("Received Secure Response from PP (Transposed)")
                    logger.debug("\n%s", large_matrix_to_string(rxSR))

                    rxSR = rxSR.T
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

            #print("Time optimal finding",time.time() - startOptimalFinding)
        
        elif (rank != 0):           
            # Perform the secure Sharing of the splitting matrix
            privateSM = qDataBase.get_merged_splitting_matrix()
            logger.info("Merged splitting options from all features and obtain the private splitting matrix with shape of {}".format(str(privateSM.shape)))
            #logger.debug("Value of the private splitting matrix is \n{}".format(str(privateSM))) 

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

                logger.debug("The private SM (transposed)")
                logger.debug("\n%s", large_matrix_to_string(privateSM.T))

            else:
                logger.info("No splitting option feasible. Sent empty splitting matrix the active party")
            status = comm.send(secureRep, dest=PARTY_ID.ACTIVE_PARTY, tag = FEDXGBOOST_MSGID.SECURE_RESPONSE)

            # Receive the optimal splitting information
            sInfo = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = FEDXGBOOST_MSGID.OPTIMAL_SPLITTING_SELECTION)            
            logger.info("Received the Splitting Info from the active party") 
        
            
        # Post processing, final announcement (optimal splitting vector)
        if (sInfo.isValid):
            sInfo = self.fed_finalize_optimal_finding(sInfo, qDataBase, privateSM)
        
        return sInfo



   
