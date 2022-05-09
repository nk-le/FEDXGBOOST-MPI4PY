from copy import deepcopy
import numpy as np
from scipy.linalg import null_space
from common.Common import logger, rank, comm, PARTY_ID, MSG_ID, SplittingInfo
from data_structure.TreeStructure import *
from data_structure.DataBaseStructure import *
from federated_xgboost.XGBoostCommon import compute_splitting_score, XgboostLearningParam
from federated_xgboost.FLTree import FLXGBoostClassifierBase, FLPlainXGBoostTree

class FEDXGBOOST_MSGID:
    SECURE_KERNEL = 200
    SECURE_RESPONSE = 201
    OPTIMAL_SPLITTING_SELECTION = 202
    OPTIMAL_SPLITTING_INFO = 203

def secure_response(privateX, U):
    r = np.random.randint(U.shape[1])
    Z = U[:, np.random.randint(U.shape[1], size=r)]
    W = np.identity(privateX.shape[0]) - np.matmul(Z, Z.T)
    return np.matmul(W,privateX)



class FedXGBoostClassifier(FLXGBoostClassifierBase):
    def __init__(self, nTree = 3):
        trees = []
        for _ in range(nTree):
            tree = FLPlainXGBoostTree()
            trees.append(tree)
        super().__init__(trees)
        

    def assign_tree(self, nTree = 3):
        trees = []
        for _ in range(self.nTree):
            tree = VerticalFedXGBoostTree()
            trees.append(tree)
        return trees

class VerticalFedXGBoostTree(FLPlainXGBoostTree):
    def __init__(self, param: XgboostLearningParam = XgboostLearningParam()):
        super().__init__(param)

    # Child class declares the privacy optimal split finding
    def fed_optimal_split_finding(self, qDataBase: QuantiledDataBase):
        # Start finding the optimal candidate federatedly
        nprocs = comm.Get_size()
        sInfo = SplittingInfo()
        privateSM = np.array([])

        if rank == PARTY_ID.ACTIVE_PARTY:
            # Perform the QR Decomposition
            matGH = np.concatenate((qDataBase.gradVec, qDataBase.hessVec), axis=1)

            #print(matGH)
            #q, r = np.linalg.qr(matGH)
            Z = null_space(matGH.T)
            logger.info("Performed QR decomposition of [G, H])")
            logger.debug("GradHess matrix: %s", str(matGH.T))
            logger.debug("Nullspace matrix: %s", str(Z))
            
            for partners in range(2, nprocs):   
                # Send the Secure kernel to the PP
                status = comm.send(Z, dest = partners, tag = FEDXGBOOST_MSGID.SECURE_KERNEL)
                #logger.warning("Sent the splitting matrix to the active party")  

                # Receive the Secure Response from the PP
                rxSR = comm.recv(source=partners, tag = FEDXGBOOST_MSGID.SECURE_RESPONSE)
                logger.warning("Received the secure response from the passive party")

                # Collect all private splitting info from the partners to find the optimal splitting candidates
                # Find the optimal splitting score iff the splitting matrix is provided
                rxSR = rxSR.T
                if rxSR.size:
                    sumGRVec = np.matmul(rxSR, qDataBase.gradVec).reshape(-1)
                    sumHRVec = np.matmul(rxSR, qDataBase.hessVec).reshape(-1)
                    sumGLVec = sum(qDataBase.gradVec) - sumGRVec
                    sumHLVec = sum(qDataBase.hessVec) - sumHRVec
                    L = compute_splitting_score(rxSR, qDataBase.gradVec, qDataBase.hessVec)
                    logger.debug("Received SM from party {} and computed:  \n".format(partners) + \
                        "GR: " + str(sumGRVec.T) + "\n" + "HR: " + str(sumHRVec.T) +\
                        "\nGL: " + str(sumGLVec.T) + "\n" + "HL: " + str(sumHLVec.T) +\
                        "\nSplitting Score: {}".format(L.T))       
                    
                    bestSplitId = np.argmax(L)
                    maxScore = L[bestSplitId]

                    # Select the optimal over all partner parties
                    if (maxScore > sInfo.bestSplitScore):
                        sInfo.bestSplitScore = maxScore
                        sInfo.bestSplitParty = partners
                        sInfo.selectedCandidate = bestSplitId
                        sInfo.bestSplittingVector = None # Unknown for AP
                                
            # Build Tree from the feature with the optimal index
            for partners in range(2, nprocs):
                data = comm.send(sInfo, dest = partners, tag = FEDXGBOOST_MSGID.OPTIMAL_SPLITTING_SELECTION)

        elif (rank != 0):           
            # Perform the secure Sharing of the splitting matrix
            privateSM = qDataBase.get_merged_splitting_matrix()
            logger.info("Merged splitting options from all features and obtain the private splitting matrix with shape of {}".format(str(privateSM.shape)))
            logger.debug("Value of the private splitting matrix is \n{}".format(str(privateSM))) 

            # Receive the secured kernel
            secureKernel = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = FEDXGBOOST_MSGID.SECURE_KERNEL)
            logger.warning("Received the secure kernel from the active party")

            # Compute the secure response and send to the active party
            secureRep = np.array([])
            if(privateSM.size):
                secureRep = secure_response(privateSM.T, secureKernel)
                logger.warning("Sent the secure response to the active party")
            else:
                logger.warning("No splitting option feasible. Sent empty splitting matrix the active party")
            status = comm.send(secureRep, dest=PARTY_ID.ACTIVE_PARTY, tag = FEDXGBOOST_MSGID.SECURE_RESPONSE)

            # Receive the optimal splitting information
            sInfo = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = FEDXGBOOST_MSGID.OPTIMAL_SPLITTING_SELECTION)            
            logger.warning("Received the Splitting Info from the active party") 
        
            
        # Post processing, final announcement (optimal splitting vector)
        sInfo = self.fed_finalize_optimal_finding(sInfo, qDataBase, privateSM)
        return sInfo



    



# class FedXGBoostSecureHandler:
#     QR = []
#     pass

#     def generate_secure_kernel(mat):
#         import scipy.linalg
#         return scipy.linalg.qr(mat)

#     def calc_secure_response(privateMat, rxKernelMat):
#         n = len(rxKernelMat) # n users 
#         r = len(rxKernelMat[0]) # number of kernel vectors
#         Z = rxKernelMat
#         return np.matmul((np.identity(n) - np.matmul(Z, np.transpose(Z))), privateMat)

#     def generate_splitting_matrix(dataVector, quantileBin):
#         n = len(dataVector) # Rows as n users
#         l = len(quantileBin) # amount of splitting candidates

#         retSplittingMat = []
#         for candidateIter in range(l):
#             v = np.zeros(n)
#             for userIter in range(n):
#                 v[userIter] = (dataVector[userIter] > max(quantileBin[candidateIter]))  
#             retSplittingMat.append(v)

#         return retSplittingMat


