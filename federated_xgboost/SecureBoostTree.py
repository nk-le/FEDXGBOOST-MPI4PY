from copy import deepcopy
import pyclbr
import sys
from typing import Any
import numpy as np
from scipy.linalg import null_space
from config import logger, rank, comm
from data_structure.TreeStructure import *
from data_structure.DataBaseStructure import *
from federated_xgboost.XGBoostCommon import compute_splitting_score, get_splitting_score, XgboostLearningParam, PARTY_ID, SplittingInfo
from federated_xgboost.FLTree import FLPlainXGBoostTree, FLXGBoostClassifierBase
from mpi4py import MPI
# Addthe external package for homomorphic encryption scheme
from Pyfhel import Pyfhel, PyPtxt, PyCtxt

import phe
from phe import paillier

class SECUREBOOST_MSGID:
    ENCRYP_GRADIENT = 100
    ENCRYP_HESSIAN = 101
    ENCRYP_AGGR_GL = 102
    ENCRYP_AGGR_HL = 103
    OPTIMAL_SPLITTING_SELECTION = 104
    OPTIMAL_SPLITTING_INFO = 105
    RAW_SPLITTING_MATRIX = 106


class SecureBoostClassifier(FLXGBoostClassifierBase):
    def __init__(self, nTree = 3):
        trees = []
        for i in range(nTree):
            tree = VerticalSecureBoostTree(i)
            trees.append(tree)
        super().__init__(trees)
        

class PseudoSecureBoostClassifier(FLXGBoostClassifierBase):
    def __init__(self, nTree = 3):
        trees = []
        for i in range(nTree):
            tree = PseudoVerticalSecureBoostTree(i)
            trees.append(tree)
        super().__init__(trees)
        



class PseudoVerticalSecureBoostTree(FLPlainXGBoostTree):
    """
    The Federated Learning XGBoost Tree applying homomorphic encryption
    """
    def __init__(self, id, param: XgboostLearningParam = XgboostLearningParam()):
        super().__init__(id, param)

        self.HEHandler = Pyfhel()  
        self.HEHandler.contextGen(p=65537, m=1024, base=3, flagBatching=True)   # Generating context. 
        self.HEHandler.keyGen()

        self.pubkey, self.privkey = paillier.generate_paillier_keypair(n_length=2048)

    # Child class declares the privacy optimal split finding
    def fed_optimal_split_finding(self, qDataBase: QuantiledDataBase):
        
        sInfo = super().fed_optimal_split_finding(qDataBase)

        """
        Pseudo encrypt to compute the total amount of the transmitted bytes
        """
        if rank == PARTY_ID.ACTIVE_PARTY:
            # Encrypt the gradient and the hessians - This task is also time expensive
            #encGArr = np.empty(qDataBase.nUsers,dtype=PyCtxt)
            #encHArr = np.empty(qDataBase.nUsers,dtype=PyCtxt)
            # Encryption takes to long. Comment for measuuring the bytes only
            # for i in range((qDataBase.nUsers)):
            #    encGArr[i] = self.HEHandler.encryptFrac(qDataBase.gradVec[i])
            #    encHArr[i] = self.HEHandler.encryptFrac(qDataBase.hessVec[i])
            
            encrG = self.HEHandler.encryptFrac(qDataBase.gradVec[0])
            
            encGPHE = self.pubkey.encrypt(0, precision = 16)
            print((encGPHE.ciphertext()))

            nTx = 2 * qDataBase.nUsers * len(encrG.to_bytes()) # Sending g and h --> multiplies by 2, Rx is neglectable
            for i in range(2, comm.Get_size()):
                self.commLogger.log_nTx(nTx, i, self.treeID)
        else:

            pass

        return sInfo

 


class VerticalSecureBoostTreePHE(FLPlainXGBoostTree):
    """
    The Federated Learning XGBoost Tree applying homomorphic encryption
    """
    def __init__(self, param: XgboostLearningParam = XgboostLearningParam()):
        super().__init__(param)
        
        self.pubkey, self.privkey = paillier.generate_paillier_keypair(n_length=2048)


    # Child class declares the privacy optimal split finding
    def fed_optimal_split_finding(self, qDataBase: QuantiledDataBase):
        # Start finding the optimal candidate federatedly
        nprocs = comm.Get_size()
        sInfo = SplittingInfo()
            
        if rank == PARTY_ID.ACTIVE_PARTY:
            # Encrypt the gradient and the hessians
            #print(qDataBase.gradVec, qDataBase.hessVec)
            encGArr = [Any for x in range(qDataBase.nUsers)]
            encHArr = [Any for x in range(qDataBase.nUsers)]
            for i in range(qDataBase.nUsers):
                qDataBase.gradVec[i]
                encGArr[i] = self.pubkey.encrypt(0, precision = 16)
                encHArr[i] = self.pubkey.encrypt(0, precision = 16)

            tmp = encGArr[0].ciphertext()
            print(sys.getsizeof(tmp))
            #print((encGArr[0].ciphertext()))
            logger.info("Encrypted the private gradients [G, H])")
            #logger.debug("Encrypted Gradient: %s", str(encGArr.T))
            #logger.debug("Encrypted Hessians: %s", str(encHArr.T))
            G = sum(qDataBase.gradVec)
            H = sum(qDataBase.hessVec)
            for partners in range(2, nprocs):   
                # Send the Secure kernel to the PP
                """
                !!! Sending this big data cause crashing
                """

                status = comm.send(encGArr, dest = partners, tag = SECUREBOOST_MSGID.ENCRYP_GRADIENT)
                status = comm.send(encHArr, dest = partners, tag = SECUREBOOST_MSGID.ENCRYP_HESSIAN)

                # Receive the Secure Response from the PP
                #rxStat = MPI.Status()
                rxGL = comm.recv(source=partners, tag = SECUREBOOST_MSGID.ENCRYP_AGGR_GL)
                rxHL = comm.recv(source=partners, tag = SECUREBOOST_MSGID.ENCRYP_AGGR_HL)

                if rxGL.size:
                    
                    GL = [self.privkey.decrypt(x) for x in rxGL]
                    HL = [self.privkey.decrypt(x) for x in rxHL]
                    GR = G - GL
                    HR = H - HL
                    L = get_splitting_score(G,H,GL,GR,HL,HR)
                    #logger.warning("Received the encrypted aggregated data from the passive party")

                    # Collect all private splitting info from the partners to find the optimal splitting candidates
                    # Find the optimal splitting score iff the splitting matrix is provided                    
                    # logger.debug("Received SM from party {} and computed:  \n".format(partners) + \
                    #     "GR: " + str(GR.T) + "\n" + "HR: " + str(HR.T) +\
                    #     "\nGL: " + str(GL.T) + "\n" + "HL: " + str(HL.T) +\
                    #     "\nSplitting Score: {}".format(L.T))       
                        
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
                data = comm.send(sInfo, dest = partners, tag = SECUREBOOST_MSGID.OPTIMAL_SPLITTING_SELECTION)
            #logger.info("Sent splitting info to clients {}".format(sInfo.bestSplitParty))

        elif (rank != 0):           
            # Perform the secure Sharing of the splitting matrix
            privateSM = qDataBase.get_merged_splitting_matrix()
            logger.info("Merged splitting options from all features and obtain the private splitting matrix with shape of {}".format(str(privateSM.shape)))
            logger.debug("Value of the private splitting matrix is \n{}".format(str(privateSM))) 

            # Receive the secured kernel
            encrG = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = SECUREBOOST_MSGID.ENCRYP_GRADIENT)
            encrH = comm.recv(bytearray(), source=PARTY_ID.ACTIVE_PARTY, tag = SECUREBOOST_MSGID.ENCRYP_HESSIAN)

            logger.warning("Received the encrypted data from the active party")
            # Aggregate the results and send to the active party
            nCandidates = privateSM.shape[0] # TODO: define a method in database class to get the amount of splititng options
            aggGL = [0 for i in range(nCandidates)]
            aggHL = [0 for i in range(nCandidates)]
            if(privateSM.size):
                # Accumulate the encrypted gradients and hessians of the left node for all possible splititing candidates
                for sc in range(nCandidates):
                    for userIndex in range(qDataBase.nUsers):
                        if(privateSM[sc, userIndex] == 0.0):
                            aggGL[sc] += encrG[userIndex]
                            aggHL[sc] += encrH[userIndex]
                    
                logger.warning("Sent the encrypted aggregate data to the active party")
            else:
                logger.warning("No splitting option feasible. Sent empty arr the active party")
            
            status = comm.send(aggGL, dest=PARTY_ID.ACTIVE_PARTY, tag = SECUREBOOST_MSGID.ENCRYP_AGGR_GL)
            status = comm.send(aggHL, dest=PARTY_ID.ACTIVE_PARTY, tag = SECUREBOOST_MSGID.ENCRYP_AGGR_HL)

            # Receive the optimal splitting information
            sInfo = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = SECUREBOOST_MSGID.OPTIMAL_SPLITTING_SELECTION)            
            logger.warning("Received the Splitting Info from the active party") 
        
        sInfo = self.fed_finalize_optimal_finding(sInfo, qDataBase, privateSM)
        return sInfo





class VerticalSecureBoostTree(FLPlainXGBoostTree):
    """
    The Federated Learning XGBoost Tree applying homomorphic encryption
    """
    def __init__(self, param: XgboostLearningParam = XgboostLearningParam()):
        super().__init__(param)

        self.HEHandler = Pyfhel()  
        self.HEHandler.contextGen(p=65537, m=1024, base=3, flagBatching=True)   # Generating context. 
        self.HEHandler.keyGen()

    # Child class declares the privacy optimal split finding
    def fed_optimal_split_finding(self, qDataBase: QuantiledDataBase):
        # Start finding the optimal candidate federatedly
        nprocs = comm.Get_size()
        sInfo = SplittingInfo()
            
        if rank == PARTY_ID.ACTIVE_PARTY:
            # Encrypt the gradient and the hessians
            encGArr = np.empty(qDataBase.nUsers,dtype=PyCtxt)
            encHArr = np.empty(qDataBase.nUsers,dtype=PyCtxt)
            for i in range((qDataBase.nUsers)):
                encGArr[i] = self.HEHandler.encryptFrac(qDataBase.gradVec[i])
                encHArr[i] = self.HEHandler.encryptFrac(qDataBase.hessVec[i])
            
            logger.info("Encrypted the private gradients [G, H])")
            logger.debug("Encrypted Gradient: %s", str(encGArr.T))
            logger.debug("Encrypted Hessians: %s", str(encHArr.T))
            for partners in range(2, nprocs):   
                # Send the Secure kernel to the PP
                status = comm.send(encGArr[0], dest = partners, tag = SECUREBOOST_MSGID.ENCRYP_GRADIENT)
                status = comm.send(encHArr[0], dest = partners, tag = SECUREBOOST_MSGID.ENCRYP_HESSIAN)

                # Receive the Secure Response from the PP
                rxGL = comm.recv(source=partners, tag = SECUREBOOST_MSGID.ENCRYP_AGGR_GL)
                rxHL = comm.recv(source=partners, tag = SECUREBOOST_MSGID.ENCRYP_AGGR_HL)

                if rxGL.size:
                    G = sum(qDataBase.gradVec)
                    H = sum(qDataBase.hessVec)
                    GL = self.HEHandler.decryptFrac(rxGL)
                    HL = self.HEHandler.decryptFrac(rxHL)
                    GR = G - GL
                    HR = H - HL
                    L = get_splitting_score(G,H,GL,GR,HL,HR)
                    logger.warning("Received the encrypted aggregated data from the passive party")

                    # Collect all private splitting info from the partners to find the optimal splitting candidates
                    # Find the optimal splitting score iff the splitting matrix is provided                    
                    logger.debug("Received SM from party {} and computed:  \n".format(partners) + \
                        "GR: " + str(GR.T) + "\n" + "HR: " + str(HR.T) +\
                        "\nGL: " + str(GL.T) + "\n" + "HL: " + str(HL.T) +\
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
                data = comm.send(sInfo, dest = partners, tag = SECUREBOOST_MSGID.OPTIMAL_SPLITTING_SELECTION)
            #logger.info("Sent splitting info to clients {}".format(sInfo.bestSplitParty))

        elif (rank != 0):           
            # Perform the secure Sharing of the splitting matrix
            privateSM = qDataBase.get_merged_splitting_matrix()
            logger.info("Merged splitting options from all features and obtain the private splitting matrix with shape of {}".format(str(privateSM.shape)))
            logger.debug("Value of the private splitting matrix is \n{}".format(str(privateSM))) 

            # Receive the secured kernel
            encrG = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = SECUREBOOST_MSGID.ENCRYP_GRADIENT)
            #print(str(encrG))


            encrH = comm.recv(bytearray(), source=PARTY_ID.ACTIVE_PARTY, tag = SECUREBOOST_MSGID.ENCRYP_HESSIAN)
            print("RxMF", str(encrG), str(encrH))

            logger.warning("Received the encrypted data from the active party")
            # Aggregate the results and send to the active party
            nCandidates = privateSM.shape[0] # TODO: define a method in database class to get the amount of splititng options
            aggGL = [] #[PyCtxt() for i in range(nCandidates)]
            aggHL = [] # for i in range(nCandidates)]
            if(privateSM.size):
                # Accumulate the encrypted gradients and hessians of the left node for all possible splititing candidates
                for sc in range(nCandidates):
                    # for userIndex in range(qDataBase.nUsers):
                    #     if(privateSM[sc, userIndex] == 0.0):
                    #         aggGL[sc] += encrG[userIndex]
                    #         aggHL[sc] += encrH[userIndex]
                    #aggGL.append(Pyfhel.add(encrG[0] + encrG[1]))
                    #print(type(encrH[0]), type(encrH[1]))
                    #aggHL.append(encrH[0] + encrH[1])
                    pass

                logger.warning("Sent the encrypted aggregate data to the active party")
            else:
                logger.warning("No splitting option feasible. Sent empty arr the active party")
            
            status = comm.send(aggGL, dest=PARTY_ID.ACTIVE_PARTY, tag = SECUREBOOST_MSGID.ENCRYP_AGGR_GL)
            status = comm.send(aggHL, dest=PARTY_ID.ACTIVE_PARTY, tag = SECUREBOOST_MSGID.ENCRYP_AGGR_HL)

            # Receive the optimal splitting information
            sInfo = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = SECUREBOOST_MSGID.OPTIMAL_SPLITTING_SELECTION)            
            logger.warning("Received the Splitting Info from the active party") 
        
        sInfo = self.fed_finalize_optimal_finding(sInfo, qDataBase, privateSM)
        return sInfo


