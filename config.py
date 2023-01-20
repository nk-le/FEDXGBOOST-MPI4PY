import numpy as np
from mpi4py import MPI
import logging
from datetime import date
import time 
import os
import random
#import matlab.engine

modelArr = ["PlainXGBoost", "FedXGBoost", "SecureBoost", "PseudoSecureBoost", "FedXGBoost-Nys"]
dataset = ["Iris", "GiveMeCredits", "Adult", "DefaultCredits", "AugData"]

CONFIG = {
  "model": modelArr[1],
  "dataset": dataset[1],
  "lambda": 1, # 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10 ###### CAI SO NAY NE
  "gamma": 0.5, # 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10  
  "MAX_DEPTH": 3,
  "MAX_TREE": 3
}

class SIM_PARAM:
  N_SAMPLE = int(8e4)
  N_FEATURE = int(10)

"""
Testing: nUsers
Dataset: GivemeCredits
N: 10k, 20k, 30k, 50k, 80k 120

"""
TEST_CASE = "VARYING_HYPERPARAM_LAMBDA_MAXDEPTH3" # NFEATURE_AugData", ACCELERATE_FEDXGBOOST_FAST_RESPONSE_SECURE


random.seed(10)
N_CLIENTS = 5
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger = logging.getLogger()
day = date.today().strftime("%b-%d-%Y")

curTime = round(time.time())

logName = 'Log/{}/{}/{}_{}_{}_{}_{}/Rank_{}.log'.format(TEST_CASE, str(day), str(curTime), str(CONFIG["dataset"]), str(CONFIG["model"]), str(CONFIG["lambda"]), str(CONFIG["gamma"]), str(rank))
#logName = 'Log/{}/{}/{}_{}_{}/Rank_{}.log'.format(TEST_CASE, str(day), str(curTime), str(CONFIG["dataset"]), str(CONFIG["model"]), str(rank))
os.makedirs(os.path.dirname(logName), exist_ok=True)

file_handler = logging.FileHandler(logName, mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
formatter = logging.Formatter('%(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.WARNING)


np.set_printoptions(linewidth=np.inf)
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(precision=4, suppress=True)
# mlEngine = matlab.engine.start_matlab()
# s = mlEngine.genpath('matlab_algo/receursive-nystrom')
# mlEngine.addpath(s, nargout = 0)
# s = mlEngine.genpath('matlab_algo/')
# mlEngine.addpath(s, nargout = 0)
