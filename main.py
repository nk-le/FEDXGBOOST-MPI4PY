from datetime import datetime
from statistics import mode
import pandas as pd
import numpy as np
from federated_xgboost.FLTree import PlainFedXGBoost
from federated_xgboost.FedXGBoostTree import FedXGBoostClassifier
from common.Common import PARTY_ID, rank, logger
from federated_xgboost.SecureBoostTree import SecureBoostClassifier

from data_preprocessing import *


def test_iris(model):
    X_train, y_train, X_test, y_test, fName = get_iris()

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

    X_train_A = X_train[:, :2]
    fNameA = fName[:2]
    #print(X_train_A)

    X_train_B = X_train[:, 2:4]
    fNameB = fName[2:4]

    X_train_C = X_train[:, 4:7]
    fNameC = fName[4:7]

    X_train_D = X_train[:, 7:]
    fNameD = fName[7:]
    #print(np.shape(fNameD), np.shape(X_train_D))

    X_test_A = X_test[:, :2]
    X_test_B = X_test[:, 2:4]
    X_test_C = X_test[:, 4:7]
    X_test_D = X_test[:, 7:]

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


def test_default_credit_client(model):
    X_train, y_train, X_test, y_test, fName = get_default_credit_client()

    X_train_A = X_train[:, 0:2]
    fNameA = fName[0:2]

    X_train_B = X_train[:, 2:5]
    fNameB = fName[2:5]

    X_train_C = X_train[:, 5:14]
    fNameC = fName[5:14]

    X_train_D = X_train[:, 14:]
    fNameD = fName[14:]

    X_test_A = X_test[:, 0:2]
    X_test_B = X_test[:, 2:5]
    X_test_C = X_test[:, 5:14]
    X_test_D = X_test[:, 14:]

    if rank == 1:
        model.append_data(X_train_A)
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

from sklearn import metrics
import sys
try:
    import logging
    #np.set_printoptions(threshold=sys.maxsize)
    
    logger.setLevel(logging.INFO)

    # Model selection
    #model = SecureBoostClassifier()
    #model = FedXGBoostClassifier()
    model = PlainFedXGBoost(10)

    # Dataset selection    
    if rank != 0:
        y_pred, y_test, model = test_default_credit_client(model)
        #y_pred, y_test, model = test_give_me_credits(model)
        #y_pred, y_test, model = test_iris(model)
        y_pred_org = y_pred.copy()
        if rank == PARTY_ID.ACTIVE_PARTY:
            y_pred = 1.0 / (1.0 + np.exp(-y_pred)) # Mapping to -1, 1
            y_pred_true = y_pred.copy()
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
            result = y_pred - y_test
            
            print("Prediction Acc: ", np.sum(result == 0) / y_pred.shape[0])
            strPred = ""
            for i in range(len(y_pred)):
                strPred += "{} -> {} >< {} \n".format(y_pred_org[i], y_pred_true[i], y_test[i])
            logger.info("Pred: %s", str(strPred))
            
            auc = metrics.roc_auc_score(y_test, y_pred_true)
            print("AUC", auc)

        model.performanceLogger.print_info()

except Exception as e:
    logger.error("Exception occurred", exc_info=True)
    print("Rank ", rank, e)
