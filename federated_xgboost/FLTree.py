import numpy as np
from common.Common import Direction, FedDirRequestInfo, FedDirResponseInfo, logger, rank, comm, PARTY_ID, MSG_ID, TreeNodeType, SplittingInfo
from algo.LossFunction import LeastSquareLoss, LogLoss
from data_structure.TreeStructure import *
from data_structure.DataBaseStructure import *
from visualizer.TreeRender import FLVisNode


from federated_xgboost.XGBoostCommon import XgboostLearningParam, compute_splitting_score

class FLPlainXGBoostTree():
    def __init__(self, param:XgboostLearningParam = XgboostLearningParam()):
        self.learningParam = param
        self.root = FLTreeNode()
        self.nNode = 0

    def fit_fed(self, y, yPred, treeID, qDataBase: QuantiledDataBase):
        logger.info("Tree is growing column-wise. Current column: %d", treeID)

        #super().fit(y_and_pred, treeID)

        """
        This function computes the gradient and the hessian vectors to perform the tree construction
        """
        # Compute the gradients and hessians
        if rank == PARTY_ID.ACTIVE_PARTY: # Calculate gradients on the node who have labels.
            G = np.array(self.learningParam.LOSS_FUNC.gradient(y, yPred)).reshape(-1)
            H = np.array(self.learningParam.LOSS_FUNC.hess(y, yPred)).reshape(-1)
            logger.info("Computed Gradients and Hessians ")
            logger.debug("G {}".format(' '.join(map(str, G))))
            logger.debug("H {}".format(' '.join(map(str, H))))

            # nprocs = comm.Get_size()
            # for partners in range(2, nprocs):   
            #     data = comm.send(G, dest = partners, tag = MSG_ID.MASKED_GH)
            #     logger.info("Sent G, H to party %d", partners)         

            qDataBase.appendGradientsHessian(G, H) 

        else:
            # data = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag=MSG_ID.MASKED_GH)
            # logger.info("Received G, H from the active party")

            dummyG = np.zeros((qDataBase.nUsers,1))
            dummyH = np.zeros((qDataBase.nUsers,1))
            qDataBase.appendGradientsHessian(dummyG, dummyH)

        if(rank != 0):
            rootNode = FLTreeNode()
            self.fed_grow(qDataBase, depth = 0, NodeDirection = TreeNodeType.ROOT, currentNode = rootNode)
            self.root = rootNode
            # Display the tree in the log file
            b = FLVisNode(self.root)
            b.display(treeID)

    def generate_leaf(self, gVec, hVec, lamb = 0.1):
        gI = sum(gVec) 
        hI = sum(hVec)
        ret = TreeNode(-1.0 * gI / (hI + lamb), leftBranch= None, rightBranch= None)
        return ret

    # This method requires normally the highest privacy concern
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


    def fed_grow(self, qDataBase: QuantiledDataBase, depth=0, NodeDirection = TreeNodeType.ROOT, currentNode : FLTreeNode = None):
        logger.info("Tree is growing depth-wise. Current depth: {}".format(depth) + " Node's type: {}".format(NodeDirection))
        currentNode.nUsers = qDataBase.nUsers

        # Assign the unique fed tree id for each node
        currentNode.FID = self.nNode
        self.nNode += 1

        # Distributed splitting evaluation

        sInfo = self.fed_optimal_split_finding(qDataBase)
        
        # Set the optimal split as the owner ID of the current tree node
        # If the selected party is me
        if(rank == sInfo.bestSplitParty):
            feature, value = qDataBase.find_fId_and_scId(sInfo.bestSplittingVector)
            sInfo.featureName = feature
            sInfo.splitValue = value
            currentNode.set_splitting_info(sInfo)
        elif rank != 0:
            currentNode.set_splitting_info(sInfo)
        sInfo.log()

        # Get the optimal splitting candidates and partition them into two databases
        if(sInfo.bestSplittingVector is not None):      
            maxDepth = 3
            # Construct the new tree if the gain is positive
            if (depth <= maxDepth) and (sInfo.bestSplitScore > 0):
                depth += 1
                lD, rD = qDataBase.partition(sInfo.bestSplittingVector)
                logger.info("Growing to the next depth %d. Splitting database into two quantiled databases", depth)
                logger.info("\nOriginal database: %s", qDataBase.get_info_string())
                logger.info("\nLeft splitted database: %s", lD.get_info_string())
                logger.info("\nRight splitted database: %s \n", rD.get_info_string())

                currentNode.leftBranch = FLTreeNode()
                currentNode.rightBranch = FLTreeNode()

                # grow recursively
                self.fed_grow(lD, depth,NodeDirection = TreeNodeType.LEFT, currentNode=currentNode.leftBranch)
                self.fed_grow(rD, depth, NodeDirection = TreeNodeType.RIGHT, currentNode=currentNode.rightBranch)
            
            else:
                endNode = self.generate_leaf(qDataBase.gradVec, qDataBase.hessVec, lamb = 0.2)
                currentNode.weight = endNode.weight

                logger.warning("Reached max-depth or Gain is negative. Terminate the tree growing,\
                     generate the leaf with weight Leaf Weight: %f", currentNode.weight)
        else:
            endNode = self.generate_leaf(qDataBase.gradVec, qDataBase.hessVec, lamb = 0.2)
            currentNode.weight = endNode.weight

            logger.warning("Splitting candidate is not feasible. Terminate the tree growing,\
                    generate the leaf with weight Leaf Weight: %f", currentNode.weight)
            
        # Post processing
        # Remove the feature for the next iteration because this is already used
        if(rank == sInfo.bestSplitParty):
            #qDataBase.remove_feature(feature) # TODO: Implement a generic method to update the database before going into the next depth
            pass

    def fed_predict(self, database: DataBase): # Encapsulated for many data
        """
        Data matrix has the same format as the data appended to the database, includes the features' values
        
        """
        myRes = np.zeros(database.nUsers, dtype=float)

        # Perform prediction for users with [idUser] --> [left, right, nextParty]
        if rank is PARTY_ID.ACTIVE_PARTY:
            # Initialize the inferrence process --> Mimic the clien service behaviour. TODO: use mpi4py standard?
            nprocs = comm.Get_size()
            for partners in range(2, nprocs):
                data = comm.send(np.zeros([0]), dest = partners, tag = MSG_ID.INIT_INFERENCE_SIG)
            logger.debug("Sent the initial inference request to all partner party.")

            # Synchronous federated Inference
            for i in range(database.nUsers):
                myRes[i] = self.classify_fed(database, userId = i)

            # Finish inference --> sending abort signal
            for partners in range(2, nprocs):
                status = comm.send(np.zeros([0]), dest = partners, tag = MSG_ID.ABORT_INFERENCE_SIG)
            logger.debug("Sent the abort inference request to all partner party.")
            
        elif rank != 0:
            # Receive sync request from the active party
            info = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.INIT_INFERENCE_SIG)
            logger.warning("Received the inital inference request from the active party. Start performing federated inferring ...") 

            # Synchronous federated Inference
            abortSig = comm.irecv(source=PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.ABORT_INFERENCE_SIG)            
            isAbort, mes = abortSig.test()
            # Synchronous modes as performing the federated inference
            while(not isAbort):
                self.classify_fed(database)        
                isAbort, mes = abortSig.test()
            
            # Synchronous federated Inference
            logger.debug("Finished federated inference!") 
        
        myRes = np.array(myRes).reshape(-1,1)
        return myRes
    
    # TODO: bring the userIdList and data, fName to preprocessing --> classifying will predict a DataBase
    def classify_fed(self, database: DataBase, userId = None):
        """
        Federated Infering
        """    
        if rank is PARTY_ID.ACTIVE_PARTY:
            # Initialize searching from the root
            curNode = self.root
            # Iterate until we find the right leaf node  
            while(not curNode.is_leaf()):                
                # Federate finding the direction for the next node
                req = FedDirRequestInfo(userId)
                req.nodeFedId = curNode.FID
                
                logger.debug("Sent the direction request to all partner party")
                status = comm.send(req, dest = curNode.owner, tag = MSG_ID.REQUEST_DIRECTION)
                
                # Receive the response
                logger.debug("Waiting for the direction response from party %d.", curNode.owner)
                dirResp = comm.recv(source = curNode.owner, tag = MSG_ID.RESPONSE_DIRECTION)

                logger.debug("Received the classification info from party %d", curNode.owner)
                logger.debug("User ID: %d| Direction: %s", (userId), str(dirResp.Direction))
                if(dirResp.Direction == Direction.LEFT):
                    curNode =curNode.leftBranch
                elif(dirResp.Direction == Direction.RIGHT):
                    curNode = curNode.rightBranch

            # Return the weight of the terminated tree leaf
            return curNode.weight

        elif rank != 0:
            # Waiting for the request from the host to return the direction
            isRxRequest = comm.iprobe(source=PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.REQUEST_DIRECTION)
            if(isRxRequest):
                logger.debug("Received the direction inference request. Start Classifying ...") 
                rxReqData = comm.recv(source = PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.REQUEST_DIRECTION)
                userClassified = rxReqData.userIdList
                rxReqData.log()
                fedNodePtr = self.root.find_child_node(rxReqData.nodeFedId)
                # Find the node and verify that it exists 
                if fedNodePtr:
                    #fedNodePtr.splittingInfo.log()
                    #print(userClassified)
                    pass
                # Classify the user according to the current node
                rep = FedDirResponseInfo(userClassified)
                # Reply the direction 
                rep.Direction = \
                    (database.featureDict[fedNodePtr.splittingInfo.featureName].data[userClassified] > fedNodePtr.splittingInfo.splitValue)
                logger.debug("User: %d, Val: %f, Thres: %f, Dir: %d", userClassified, \
                    database.featureDict[fedNodePtr.splittingInfo.featureName].data[userClassified], fedNodePtr.splittingInfo.splitValue, rep.Direction)                    
                #print(rep.Direction)
                #rep.Direction = Direction.DEFAULT
                assert rep.Direction != Direction.DEFAULT, "Invalid classification"

                # Transfer back to the active party
                status = comm.send(rep, dest = PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.RESPONSE_DIRECTION)
                logger.debug("Received the request. Classify users in direction %s. Sent to active party.", str(rep.Direction)) 
            else:
                #logger.info("Pending...")
                pass
            return 0

    

    def predict(self, dataTable, featureName):        
        dataBase = DataBase.data_matrix_to_database(dataTable, featureName)
        return self.fed_predict(dataBase)
     