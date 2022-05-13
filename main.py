from datetime import datetime
from pyexpat import model
from statistics import mode
import mpi4py
import pandas as pd
import numpy as np
from algo.LossFunction import LeastSquareLoss, LogLoss
from data_structure.DataBaseStructure import QuantileParam
from federated_xgboost.FLTree import PlainFedXGBoost
from federated_xgboost.FedXGBoostTree import FedXGBoostClassifier
from config import rank, logger, comm
from federated_xgboost.SecureBoostTree import SecureBoostClassifier

from data_preprocessing import *
from federated_xgboost.XGBoostCommon import XgboostLearningParam, PARTY_ID 


def pre_config():
    XgboostLearningParam.LOSS_FUNC = LogLoss()
    XgboostLearningParam.GAMMA = 0.5
    XgboostLearningParam.LAMBDA = 1
    XgboostLearningParam.MAX_DEPTH = 3
    XgboostLearningParam.N_TREES = 5
    QuantileParam.epsilon = QuantileParam.epsilon
    QuantileParam.thres_balance = 0.1
    SIM_PARAM.N_SAMPLE = 3e4
   

def log_distribution(X_train, y_train, y_test):
    nTrain = len(y_train)
    nZeroTrain = np.count_nonzero(y_train == 0)
    rTrain = nZeroTrain/nTrain

    nTest = len(y_test)
    nZeroTest = np.count_nonzero(y_test == 0)
    rTest = nZeroTest / nTest
    logger.warning("DataDistribution, nTrain: %d, zeroRate: %f, nTest: %d, ratioTest: %f, nFeature: %d", 
    nTrain, rTrain, nTest, rTest, X_train.shape[1])


def test_iris(model):
    X_train, y_train, X_test, y_test, fName = get_iris()
    log_distribution(X_train, y_train, y_test)

    X_train_A = X_train[:, 0].reshape(-1, 1)
    fNameA = fName[0]

    X_train_B = X_train[:, 2].reshape(-1, 1)
    fNameB = fName[2]
    X_train_C = X_train[:, 1].reshape(-1, 1)
    fNameC = fName[1]
    X_train_D = X_train[:, 3].reshape(-1, 1)
    fNameD = fName[3]
    X_test_A = X_test[:, 0].reshape(-1, 1)
    X_test_B = X_test[:, 2].reshape(-1, 1)
    X_test_C = X_test[:, 1].reshape(-1, 1)
    X_test_D = X_test[:, 3].reshape(-1, 1)

    if rank == 1:
        model.append_data(X_train_A, fNameA)
        model.append_label(y_train)
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_B, fNameB)
        model.append_label(np.zeros_like(y_train))
    elif rank == 3:
        model.append_data(X_train_C, fNameC)
        model.append_label(np.zeros_like(y_train))
    elif rank == 4:
        model.append_data(X_train_D, fNameD)
        model.append_label(np.zeros_like(y_train))
    else:
        model.append_data(X_train_A)
        model.append_label(np.zeros_like(y_train))

    model.print_info()
    model.boost()
    
    if rank == 1:
        y_pred = model.predict(X_test_A, fNameA)
    elif rank == 2:
        y_pred = model.predict(X_test_B, fNameB)
    elif rank == 3:
        y_pred = model.predict(X_test_C, fNameC)
    elif rank == 4:
        y_pred = model.predict(X_test_D, fNameD)
    else:
        model.predict(np.zeros_like(X_test_A))

    y_pred_org = y_pred.copy()

    return y_pred_org, y_test, model

def test_give_me_credits(model):
    X_train, y_train, X_test, y_test, fName = get_give_me_credits()
    log_distribution(X_train, y_train, y_test)

    X_train_A = X_train[:, :2]
    fNameA = fName[:2]
    X_test_A = X_test[:, :2]

    #print(X_train_A)

    X_train_B = X_train[:, 2:]
    fNameB = fName[2:]
    X_test_B = X_test[:, 2:]

     # np.concatenate((X_train_A, y_train))
    if rank == 1:
        #print("Test A", len(X_train_A), len(X_train_A[0]), len(y_train), len(y_train[0]))
        #print("Test A", X_train_A.shape[0], len(X_train_A[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_A, fNameA)
        model.append_label(y_train)
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_B, fNameB)
        model.append_label(np.zeros_like(y_train))
    else:
        model.append_data(X_train_A)
        model.append_label(np.zeros_like(y_train))


    model.print_info()


    model.boost()

    if rank == 1:
        y_pred = model.predict(X_test_A, fNameA)
    elif rank == 2:
        y_pred = model.predict(X_test_B, fNameB)
    else:
        model.predict(np.zeros_like(X_test_A))

    y_pred_org = y_pred.copy()
   
    return y_pred_org, y_test, model


def test_default_credit_client(model):
    X_train, y_train, X_test, y_test, fName = get_default_credit_client()
    log_distribution(X_train, y_train, y_test)

    X_train_A = X_train[:, 0:2]
    fNameA = fName[0:2]
    X_test_A = X_test[:, 0:2]

    X_train_B = X_train[:, 2:]
    fNameB = fName[2:]
    X_test_B = X_test[:, 2:]


    if rank == 1:
        model.append_data(X_train_A, fNameA)
        model.append_label(y_train)
    elif rank == 2:
        model.append_data(X_train_B, fNameB)
        model.append_label(np.zeros_like(y_train))
    else:
        model.append_data(X_train_A)
        model.append_label(np.zeros_like(y_train))


    model.print_info()
    model.boost()

    if rank == 1:
        y_pred = model.predict(X_test_A, fNameA)
    elif rank == 2:
        y_pred = model.predict(X_test_B, fNameB)
    else:
        model.predict(np.zeros_like(X_test_A))

    y_pred_org = y_pred.copy()
    
    return y_pred_org, y_test, model


def test_adult(model):

    X_train, y_train, X_test, y_test, segment_A, segment_B, segment_C = get_adults()
    log_distribution(X_train, y_train, y_test)
    
    X_train_A = X_train[:, 0:2]
    X_train_B = X_train[:, 2:]


    X_test_A = X_test[:, 0:2]
    X_test_B = X_test[:, 2:]

    if rank == 1:
        model.append_data(X_train_A)
        model.append_label(y_train)
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.append_data(X_train_B)
        model.append_label(np.zeros_like(y_train))
    else:
        model.append_data(X_train_A)
        model.append_label(np.zeros_like(y_train))


    model.print_info()
    model.boost()

    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:
        y_pred = model.predict(X_test_B)
    else:
        model.predict(np.zeros_like(X_test_A))

    y_pred_org = y_pred.copy()
    
    return y_pred_org, y_test, model

from sklearn import metrics
import sys
from config import CONFIG, dataset


def main():
    try:
        pre_config()

        import logging
        #np.set_printoptions(threshold=sys.maxsize)
        
        logger.setLevel(logging.WARNING)

        # Model selection
        if CONFIG["model"] == "PlainXGBoost":
            model = PlainFedXGBoost(XgboostLearningParam.N_TREES)
        elif CONFIG["model"] == "FedXGBoost":
            model = FedXGBoostClassifier(XgboostLearningParam.N_TREES)
        elif CONFIG["model"] == "SecureBoost": 
            model = SecureBoostClassifier(XgboostLearningParam.N_TREES)

        # Log the test case and the parameters
        logger.warning("TestInfo, {0}".format(CONFIG))
        logger.warning("XGBoostParameter, nTree: %d, maxDepth: %d, lambda: %f, gamma: %f", 
        XgboostLearningParam.N_TREES, XgboostLearningParam.MAX_DEPTH, XgboostLearningParam.LAMBDA, XgboostLearningParam.GAMMA)
        logger.warning("QuantileParameter, eps: %f, thres: %f", QuantileParam.epsilon, QuantileParam.thres_balance)

        # Dataset selection    
        if rank != 0:
            if CONFIG["dataset"] == dataset[0]:
                y_pred, y_test, model = test_iris(model)
            elif CONFIG["dataset"] == dataset[1]:
                y_pred, y_test, model = test_give_me_credits(model)
            elif CONFIG["dataset"] == dataset[2]:
                y_pred, y_test, model = test_adult(model)
            elif CONFIG["dataset"] == dataset[3]:
                y_pred, y_test, model = test_default_credit_client(model)
            
            if rank == PARTY_ID.ACTIVE_PARTY:
                acc, auc = model.evaluate(y_pred, y_test, treeid="FINAL")

            
                
                print("Prediction: ", acc, auc)
                #strPred = ""
                # for i in range(len(y_pred)):
                #     strPred += "{} -> {} >< {} \n".format(y_pred_org[i], y_pred_true[i], y_test[i])
                # logger.debug("Pred: %s", str(strPred))
                
                
                model.log_info()

    except Exception as e:
        logger.error("Exception occurred", exc_info=True)
        print("Rank ", rank, e)




main()

def automated():
    """
    Currently not using this because the data synchronization is not yet verified.
    """
    modelArr = ["PlainXGBoost", "FedXGBoost", "SecureBoost"]
    dataset = ["Iris", "GiveMeCredits", "Adult", "DefaultCredits"]

    try:
        for i in range (len(modelArr)):
            CONFIG["model"] = modelArr[i]
            for j in range(len(dataset)):
                CONFIG["dataset"] = dataset[j]
                print("Rank", rank, "Testing", CONFIG["model"], CONFIG["dataset"])

    except Exception as e:
            logger.error("Exception occurred", exc_info=True)
            print("Rank ", rank, e)
        