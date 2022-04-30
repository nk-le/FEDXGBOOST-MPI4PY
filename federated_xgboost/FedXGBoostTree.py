import numpy as np
from common.Common import logger, rank, comm, PARTY_ID, MSG_ID, SplittingInfo
from data_structure.TreeStructure import *
from data_structure.DataBaseStructure import *
from federated_xgboost.XGBoostCommon import compute_splitting_score, XgboostLearningParam
from federated_xgboost.FLTree import FLPlainXGBoostTree

class VerticalFedXGBoostTree(FLPlainXGBoostTree):
    def __init__(self, param: XgboostLearningParam = XgboostLearningParam()):
        super().__init__(param)

    # Child class declares the privacy optimal split finding
    def fed_optimal_split_finding(self, qDataBase: QuantiledDataBase):
        # Start finding the optimal candidate federatedly
        if rank == PARTY_ID.ACTIVE_PARTY:
            sInfo = SplittingInfo()
            nprocs = comm.Get_size()
            # Collect all private splitting info from the partners to find the optimal splitting candidates
            for partners in range(2, nprocs):   
                rxSM = comm.recv(source = partners, tag = MSG_ID.RAW_SPLITTING_MATRIX)

                # Find the optimal splitting score iff the splitting matrix is provided
                if rxSM.size:
                    sumGRVec = np.matmul(rxSM, qDataBase.gradVec).reshape(-1)
                    sumHRVec = np.matmul(rxSM, qDataBase.hessVec).reshape(-1)
                    sumGLVec = sum(qDataBase.gradVec) - sumGRVec
                    sumHLVec = sum(qDataBase.hessVec) - sumHRVec
                    L = compute_splitting_score(rxSM, qDataBase.gradVec, qDataBase.hessVec, 0.01)

                    logger.debug("Received SM from party {} and computed:  \n".format(partners) + \
                        "GR: " + str(sumGRVec.T) + "\n" + "HR: " + str(sumHRVec.T) +\
                        "\nGL: " + str(sumGLVec.T) + "\n" + "HL: " + str(sumHLVec.T) +\
                        "\nSplitting Score: {}".format(L.T))       
                    
                    # Optimal candidate of 1 partner party
                    # Select the optimal candidates without all zeros or one elements of the splitting)
                    isValid = False
                    excId = np.zeros(L.size, dtype=bool)
                    for id in range(len(L)):
                        splitVector = rxSM[id, :]

                        nL = np.count_nonzero(splitVector == 0.0)
                        nR = np.count_nonzero(splitVector == 1.0)
                        thres = 0.1 # TODO: bring this value outside as parameters 
                        isValid = (((nL/len(splitVector)) > thres) and ((nR/len(splitVector)) > thres))
                        #print(nL, nR, len(splitVector), isValid)
                        if not isValid:
                            excId[id] = True

                    # Mask the exception index to avoid strong imbalance between each node's users ratio     
                    tmpL = np.ma.array(L, mask=excId) 
                    bestSplitId = np.argmax(tmpL)
                    splitVector = rxSM[bestSplitId, :]
                    maxScore = tmpL[bestSplitId]     

                    # Select the optimal over all partner parties
                    if (maxScore > sInfo.bestSplitScore):
                        sInfo.bestSplitScore = maxScore
                        sInfo.bestSplitParty = partners
                        sInfo.selectedCandidate = bestSplitId
                        sInfo.bestSplittingVector = rxSM[bestSplitId, :]
                                
            # Build Tree from the feature with the optimal index
            for partners in range(2, nprocs):
                data = comm.send(sInfo, dest = partners, tag = MSG_ID.OPTIMAL_SPLITTING_INFO)
                logger.info("Sent splitting info to clients {}".format(partners))

        elif (rank != 0):           
            # Perform the secure Sharing of the splitting matrix
            # qDataBase.printInfo()
            privateSM = qDataBase.get_merged_splitting_matrix()
            logger.info("Merged splitting options from all features and obtain the private splitting matrix with shape of {}".format(str(privateSM.shape)))
            logger.debug("Value of the private splitting matrix is \n{}".format(str(privateSM))) 

            # Send the splitting matrix to the active party
            txSM = comm.send(privateSM, dest = PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.RAW_SPLITTING_MATRIX)
            logger.warning("Sent the splitting matrix to the active party")         

            sInfo = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.OPTIMAL_SPLITTING_INFO)
            logger.warning("Received the Splitting Info from the active party") 


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


