
import pandas as pd
import numpy as np
from common.Common import rank

def get_iris():
    data = pd.read_csv('./dataset/iris.csv').values

    zero_index = data[:, -1] == 0
    one_index = data[:, -1] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    train_size_zero = int(zero_data.shape[0] * 0.8)
    train_size_one = int(one_data.shape[0] * 0.8)
    X_train, X_test = np.concatenate((zero_data[:train_size_zero, :-1], one_data[:train_size_one, :-1]), 0), \
                      np.concatenate((zero_data[train_size_zero:, :-1], one_data[train_size_one:, :-1]), 0)
    y_train, y_test = np.concatenate((zero_data[:train_size_zero, -1].reshape(-1,1), one_data[:train_size_one, -1].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:, -1].reshape(-1, 1), one_data[train_size_one:, -1].reshape(-1, 1)), 0)

    fName = [['sepal length'],['sepal width'],['pedal length'],['pedal width']]

    return X_train, y_train, X_test, y_test, fName

def get_give_me_credits():
    data = pd.read_csv('./dataset/GiveMeSomeCredit/cs-training.csv')
    data.dropna(inplace=True)
    fName = ['SeriousDlqin2yrs',
       'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']

    
    data = data[fName].values
    
    # Normalize the data
    data = data / data.max(axis=0)

    ratio = 10000 / data.shape[0]

    zero_index = data[:, 0] == 0
    one_index = data[:, 0] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    zero_ratio = len(zero_data) / data.shape[0]
    one_ratio = len(one_data) / data.shape[0]
    num = 10000
    train_size_zero = int(zero_data.shape[0] * ratio) + 1
    train_size_one = int(one_data.shape[0] * ratio)

    if rank == 1:
        print("Data Dsitribution")
        print(zero_ratio, one_ratio)

    X_train, X_test = np.concatenate((zero_data[:train_size_zero, 1:], one_data[:train_size_one, 1:]), 0), \
                      np.concatenate((zero_data[train_size_zero:train_size_zero+int(num * zero_ratio)+1, 1:], one_data[train_size_one:train_size_one+int(num * one_ratio), 1:]), 0)
    y_train, y_test = np.concatenate(
        (zero_data[:train_size_zero, 0].reshape(-1, 1), one_data[:train_size_one, 0].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:train_size_zero+int(num * zero_ratio)+1, 0].reshape(-1, 1),
                                      one_data[train_size_one:train_size_one+int(num * one_ratio), 0].reshape(-1, 1)), 0)



    fNameNoLabel = ['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']

    return X_train, y_train, X_test, y_test, fNameNoLabel

def get_default_credit_client():
    data = pd.read_csv('./dataset/DefaultsOfCreditCardsClient/UCI_Credit_Card.csv')
    data.dropna(inplace=True)

    fName = ["ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
            "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
            "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
            "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",
            "default.payment.next.month"]

    data = data[fName].values
    
    # Normalize the data
    data = data / data.max(axis=0)

    # Get the ratio of the dataset used for training
    ratio = 1/3

    zero_index = data[:, -1] == 0
    one_index = data[:, -1] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    zero_ratio = len(zero_data) / data.shape[0]
    one_ratio = len(one_data) / data.shape[0]
    nTest = 8000
    train_size_zero = int(zero_data.shape[0] * ratio) + 1
    train_size_one = int(one_data.shape[0] * ratio)

    if rank == 1:
        print("Data Dsitribution")
        print(zero_ratio, one_ratio)

    X_train = np.concatenate((zero_data[:train_size_zero, :-1], one_data[:train_size_one, :-1]), 0)
                      
    X_test = np.concatenate((zero_data[train_size_zero:train_size_zero+int(nTest * zero_ratio)+1, :-1], 
                            one_data[train_size_one:train_size_one+int(nTest * one_ratio), :-1]), 0)
    
    y_train = np.concatenate((zero_data[:train_size_zero, -1].reshape(-1, 1), one_data[:train_size_one, -1].reshape(-1, 1)), 0)
    
    y_test = np.concatenate((zero_data[train_size_zero:train_size_zero+int(nTest * zero_ratio)+1, -1].reshape(-1, 1),
                            one_data[train_size_one:train_size_one+int(nTest* one_ratio), -1].reshape(-1, 1)), 0)



    fName = ["ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
            "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
            "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
            "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

    return X_train, y_train, X_test, y_test, fName

def get_adults():
    data = np.load('./dataset/adult.npy')
    data = data / data.max(axis=0)

    ratio = 0.8

    zero_index = data[:, 0] == 0
    one_index = data[:, 0] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]

    train_size_zero = int(zero_data.shape[0] * ratio) + 1
    train_size_one = int(one_data.shape[0] * ratio)

    X_train, X_test = np.concatenate((zero_data[:train_size_zero, 1:], one_data[:train_size_one, 1:]), 0), \
                      np.concatenate((zero_data[train_size_zero:, 1:], one_data[train_size_one:, 1:]), 0)
    y_train, y_test = np.concatenate(
        (zero_data[:train_size_zero, 0].reshape(-1, 1), one_data[:train_size_one, 0].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:, 0].reshape(-1, 1),
                                      one_data[train_size_one:, 0].reshape(-1, 1)), 0)

    segment_A = int(0.2 * (data.shape[1] - 1))
    segment_B = segment_A + int(0.2 * (data.shape[1] - 1))
    segment_C = segment_B + int(0.3 * (data.shape[1] - 1))

    return X_train, y_train, X_test, y_test, segment_A, segment_B, segment_C

    
