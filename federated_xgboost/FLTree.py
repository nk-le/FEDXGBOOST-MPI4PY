import sys
import numpy as np
from common.BasicTypes import Direction
from config import  N_CLIENTS, logger, rank, comm
from mpi4py import MPI
from algo.LossFunction import LeastSquareLoss, LogLoss
from data_structure.TreeStructure import *
from data_structure.DataBaseStructure import *
from federated_xgboost.PerformanceLogger import CommunicationLogger,TimeLogger
from visualizer.TreeRender import FLVisNode
from copy import deepcopy

from sklearn import metrics
from federated_xgboost.XGBoostCommon import XgboostLearningParam, compute_splitting_score, SplittingInfo, FedDirRequestInfo, FedDirResponseInfo, PARTY_ID

class MSG_ID:
    MASKED_GH = 99
    RAW_SPLITTING_MATRIX = 98
    OPTIMAL_SPLITTING_INFO = 97

    REQUEST_DIRECTION = 96
    RESPONSE_DIRECTION = 95
    OPTIMAL_SPLITTING_SELECTION = 94
    INIT_INFERENCE_SIG = 89
    ABORT_INFERENCE_SIG = 90
    ABORT_BOOSTING_SIG = 88
    

class FLXGBoostClassifierBase():
    def __init__(self, treeSet):
        self.nTree = len(treeSet)
        self.trees = treeSet #self.assign_tree()
        
        self.dataBase = DataBase()
        self.label = []
        self.excTimeLogger = TimeLogger()

    def log_info(self):
        self.excTimeLogger.log()
        for tree in self.trees:
            tree.commLogger.log()

    def assign_tree():
        raise NotImplementedError

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
        orgData = deepcopy(self.dataBase)
        y = self.label
        y_pred = np.zeros(np.shape(self.label))
        #y_pred = np.ones(np.shape(self.label)) * 0.5
        y_pred = np.ones(np.shape(self.label))

        
        nprocs = comm.Get_size()
        if rank == PARTY_ID.ACTIVE_PARTY:
            newTreeGain = 0
            loss = self.trees[0].learningParam.LOSS_FUNC.diff(y, y_pred)
            print("nTreeTotal", self.nTree,"Loss", abs(loss), "Tree Gain", newTreeGain)
            logger.warning("Boosting, TreeID: %d, Loss: %f, Gain: %f", -1, abs(loss), abs(newTreeGain))

        # Start federated boosting
        tStartBoost = self.excTimeLogger.log_start_boosting()
        for i in range(self.nTree): 
            tStartTree = TimeLogger.tic()    
            # Perform tree boosting
            dataFit = QuantiledDataBase(self.dataBase)
            self.trees[i].fit_fed(y, y_pred, dataFit)
            self.excTimeLogger.log_dt_fit(tStartTree, treeID=i) # Log the executed time

            tStartPred = TimeLogger.tic()
            update_pred = self.trees[i].fed_predict(orgData)
            self.excTimeLogger.log_dt_pred(tStartPred, treeID=i)


            if rank == PARTY_ID.ACTIVE_PARTY:
                update_pred = np.reshape(update_pred, (self.dataBase.nUsers, 1))
                # aggresgate the prediction to compute the loss
                y_pred += update_pred
                self.evaluatePrediction(y_pred, y, i)
                
                # Evaluation
                prevLoss = loss
                loss = self.evaluateTree(y_pred, y, i)
                
                # If terminating condition is triggered, we send the signal to all partners
                isTerminated = abs(abs(prevLoss) - abs(loss)) < XgboostLearningParam.LOSS_TERMINATE
                if isTerminated:
                    print("Sending abort boosting flags to PP")
                    logger.warning("Sending abort boosting flags to PP")
                    for partners in range(2, nprocs):
                        data = comm.send(True, dest = partners, tag = MSG_ID.ABORT_BOOSTING_SIG)
                    
                    break
                else:
                    for partners in range(2, nprocs):
                        data = comm.send(False, dest = partners, tag = MSG_ID.ABORT_BOOSTING_SIG)
 

            else:
                abortFlag = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.ABORT_BOOSTING_SIG)
                if(abortFlag):
                    print("Received the abort boosting flag from AP")
                    logger.warning("Received the abort boosting flag from AP")
                    break       

        print("Received the abort boosting flag from AP")
        self.excTimeLogger.log_end_boosting(tStartBoost)

    def evaluateTree(self, yPred, y, treeid = int):
        newTreeGain = abs(self.trees[treeid].root.compute_score())
        loss = self.trees[treeid].learningParam.LOSS_FUNC.diff(y, yPred)
        print("Loss", abs(loss), "Tree Gain", newTreeGain)
        logger.warning("Boosting, TreeID: %d, Loss: %f, Gain: %f", treeid, abs(loss), abs(newTreeGain))
        return loss

    def evaluatePrediction(self, y_pred, y, treeid = None):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred)) # Mapping to -1, 1
        y_pred_true = y_pred.copy()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        result = y_pred - y
        acc = np.sum(result == 0) / y_pred.shape[0]
        auc = metrics.roc_auc_score(y, y_pred_true)
        logger.warning("Metrics, TreeID: %s, acc: %f, auc: %f", str(treeid), acc, auc)
        

        return acc, auc

    def predict(self, X, fName = None):
        y_pred = None
        data_num = X.shape[0]
        # Make predictions
        testDataBase = DataBase.data_matrix_to_database(X, fName)
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.fed_predict(testDataBase)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred).reshape(data_num, -1)
            if rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred
        return y_pred


class PlainFedXGBoost(FLXGBoostClassifierBase):
    def __init__(self, nTree = 3):
        trees = []
        for i in range(nTree):
            tree = FLPlainXGBoostTree(i)
            trees.append(tree)
        super().__init__(trees)




class FLPlainXGBoostTree():
    def __init__(self, id, param:XgboostLearningParam = XgboostLearningParam()):
        self.learningParam = param
        self.root = FLTreeNode()
        self.nNode = 0
        self.treeID = id
        self.commLogger = CommunicationLogger(N_CLIENTS)

    def fit_fed(self, y, yPred, qDataBase: QuantiledDataBase):
        logger.info("Tree is growing column-wise. Current column: %d", self.treeID)

        """
        This function computes the gradient and the hessian vectors to perform the tree construction
        """
        # Compute the gradients and hessians
        if rank == PARTY_ID.ACTIVE_PARTY: # Calculate gradients on the node who have labels.
            G = np.array(self.learningParam.LOSS_FUNC.gradient(y, yPred)).reshape(-1)
            #G = G/ np.linalg.norm(G)
            H = np.array(self.learningParam.LOSS_FUNC.hess(y, yPred)).reshape(-1)
            #H = H/ np.linalg.norm(G)
            logger.debug("Computed Gradients and Hessians ")
            logger.debug("G {}".format(' '.join(map(str, G))))
            logger.debug("H {}".format(' '.join(map(str, H))))
            qDataBase.appendGradientsHessian(G, H) 

        else:
            dummyG = np.zeros((qDataBase.nUsers,1))
            dummyH = np.zeros((qDataBase.nUsers,1))
            qDataBase.appendGradientsHessian(dummyG, dummyH)

        # Start tree boosting
        if(rank != 0):
            rootNode = FLTreeNode()

            # All parties grow the tree distributedly
            self.fed_grow(qDataBase, depth = 1, NodeDirection = TreeNodeType.ROOT, currentNode = rootNode)
            self.root = rootNode

            # Display the tree in the log file
            b = FLVisNode(self.root)
            b.display(self.treeID)

    def fed_optimal_split_finding(self, qDataBase: QuantiledDataBase):
        # Each party studies their own user's distribution and prepare the splitting matrix
        privateSM = qDataBase.get_merged_splitting_matrix()
        sInfo = SplittingInfo()

        # Start finding the optimal candidate federatedly
        if rank == PARTY_ID.ACTIVE_PARTY:
            if privateSM.size: # check it own candidate
                score, maxScore, bestSplitId = compute_splitting_score(privateSM, qDataBase.gradVec, qDataBase.hessVec)
                if(maxScore > 0):
                    sInfo.isValid = True
                    sInfo.bestSplitParty = PARTY_ID.ACTIVE_PARTY
                    sInfo.selectedCandidate = bestSplitId
                    sInfo.bestSplitScore = maxScore
            
            nprocs = comm.Get_size()
            # Collect all private splitting info from the partners to find the optimal splitting candidates
            for i in range(2, nprocs):
                # Receive the Secure Response from the PP
                stat = MPI.Status()
                rxSM = comm.recv(source=MPI.ANY_SOURCE, tag = MSG_ID.RAW_SPLITTING_MATRIX, status = stat)
                logger.info("Received the secure response from the passive party")            
                self.commLogger.log_nRx(rxSM.size * rxSM.itemsize, stat.Get_source(), self.treeID)

                # Find the optimal splitting score iff the splitting matrix is provided
                if rxSM.size:
                    score, maxScore, bestSplitId = compute_splitting_score(rxSM, qDataBase.gradVec, qDataBase.hessVec)
                    # Select the optimal over all partner parties
                    if (maxScore > sInfo.bestSplitScore):
                        sInfo.bestSplitScore = maxScore
                        sInfo.bestSplitParty = stat.Get_source()
                        sInfo.selectedCandidate = bestSplitId
                        sInfo.isValid = True
                                
            # Build Tree from the feature with the optimal index
            # Build Tree from the feature with the optimal index
            sInfo.log()
            for partners in range(2, nprocs):
                data = comm.send(sInfo, dest = partners, tag = MSG_ID.OPTIMAL_SPLITTING_SELECTION)

        elif (rank != 0):           
            # Perform the secure Sharing of the splitting matrix
            # qDataBase.printInfo()
            
            logger.info("Merged splitting options from all features and obtain the private splitting matrix with shape of {}".format(str(privateSM.shape)))
            logger.debug("Value of the private splitting matrix is \n{}".format(str(privateSM))) 

            # Send the splitting matrix to the active party
            txSM = comm.send(privateSM, dest = PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.RAW_SPLITTING_MATRIX)
            logger.info("Sent the splitting matrix to the active party")         

            # Receive the optimal splitting information
            sInfo = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.OPTIMAL_SPLITTING_SELECTION)            
            sInfo.log()
            logger.info("Received the Splitting Info from the active party")   


            # Post processing the splitting information before returning
            # Set the optimal split as the owner ID of the current tree node
            # If the selected party is me  

        if (sInfo.isValid):
            #print(rank, sInfo.get_str_split_info())
            sInfo = self.fed_finalize_optimal_finding(sInfo, qDataBase, privateSM)
        
        return sInfo


    def fed_finalize_optimal_finding(self, sInfo: SplittingInfo, qDataBase: QuantiledDataBase, privateSM = np.array([])):
        # Set the optimal split as the owner ID of the current tree node
        # If the selected party is me
        # TODO: Considers implement this generic --> direct in the grow method as post processing?
        if(rank == sInfo.bestSplitParty):
            sInfo.bestSplittingVector = privateSM[sInfo.selectedCandidate,:]
            feature, value = qDataBase.find_fId_and_scId(sInfo.bestSplittingVector)
                
            updateSInfo = deepcopy(sInfo)
            updateSInfo.bestSplittingVector = privateSM[sInfo.selectedCandidate,:]
            nprocs = comm.Get_size()
            for partners in range(1, nprocs):
                if(partners != rank): # only send to the other parties
                    status = comm.send(updateSInfo, dest=partners, tag = MSG_ID.OPTIMAL_SPLITTING_INFO)

            # Only the selected rank has these information so it saves for itself
            sInfo.featureName = feature
            sInfo.splitValue = value

        # The other parties receive the final splitting info to construct the tree node
        elif(rank != 0):
            sInfo = comm.recv(source = sInfo.bestSplitParty, tag = MSG_ID.OPTIMAL_SPLITTING_INFO)
        
        return sInfo


    def fed_grow(self, qDataBase: QuantiledDataBase, depth, NodeDirection = TreeNodeType.ROOT, currentNode : FLTreeNode = None):
        logger.info("Tree is growing depth-wise. Current depth: {}".format(depth) + " Node's type: {}".format(NodeDirection))
        currentNode.nUsers = qDataBase.nUsers

        # Assign the unique fed tree id for each nodeand save the splitting info for each node
        currentNode.FID = self.nNode
        self.nNode += 1
        import time
        #start_time = time.time()
        sInfo = self.fed_optimal_split_finding(qDataBase)
        sInfo.log()
        currentNode.set_splitting_info(sInfo)

        # Get the optimal splitting candidates and partition them into two databases
        if(sInfo.isValid):      
            maxDepth = XgboostLearningParam.MAX_DEPTH
            # Construct the new tree if the gain is positive
            if (depth < maxDepth) and (sInfo.bestSplitScore > 0):
                depth += 1
                lD, rD = qDataBase.partition(sInfo.bestSplittingVector)
                logger.info("Splitting the database according to the best splitting vector.")
                logger.debug("\nOriginal database: %s", qDataBase.get_info_string())
                logger.debug("\nLeft splitted database: %s", lD.get_info_string())
                logger.debug("\nRight splitted database: %s \n", rD.get_info_string())

                # grow recursively
                currentNode.leftBranch = FLTreeNode()
                currentNode.rightBranch = FLTreeNode()
                self.fed_grow(lD, depth,NodeDirection = TreeNodeType.LEFT, currentNode=currentNode.leftBranch)
                self.fed_grow(rD, depth, NodeDirection = TreeNodeType.RIGHT, currentNode=currentNode.rightBranch)
            
            else:
                weight, score = FLTreeNode.compute_leaf_param(qDataBase.gradVec, qDataBase.hessVec, XgboostLearningParam.LAMBDA)
                currentNode.weight = weight
                currentNode.score = score
                currentNode.leftBranch = None
                currentNode.rightBranch = None

                logger.info("Reached max-depth or Gain is negative. Terminate the tree growing, generate the leaf with weight Leaf Weight: %f", currentNode.weight)
        else:
            weight, score = FLTreeNode.compute_leaf_param(qDataBase.gradVec, qDataBase.hessVec, XgboostLearningParam.LAMBDA)
            currentNode.weight = weight
            currentNode.score = score
            currentNode.leftBranch = None
            currentNode.rightBranch = None

            logger.info("Splitting candidate is not feasible. Terminate the tree growing and generate the leaf with weight Leaf Weight: %f", currentNode.weight)
            
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
                if(curNode.owner is PARTY_ID.ACTIVE_PARTY): # If it is me, just find the sub node myself  
                    direction = \
                    (database.featureDict[curNode.splittingInfo.featureName].data[userId] > curNode.splittingInfo.splitValue)
                    if(direction == Direction.LEFT):
                        curNode =curNode.leftBranch
                    elif(direction == Direction.RIGHT):
                        curNode = curNode.rightBranch
                else:         
                    # Federate finding the direction for the next node
                    req = FedDirRequestInfo(userId)
                    req.nodeFedId = curNode.FID
                    
                    logger.debug("Sent the direction request to all partner parties")
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
                #rxReqData.log()
                # Find the node and verify that it exists 
                fedNodePtr = self.root.find_child_node(rxReqData.nodeFedId)  
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
     