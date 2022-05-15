import numpy as np
from mpi4py import MPI
import logging
from datetime import date
import time 
import os
import matlab.engine

mlEngine = matlab.engine.start_matlab()
s = mlEngine.genpath('matlab_algo/receursive-nystrom')
mlEngine.addpath(s, nargout = 0)
s = mlEngine.genpath('matlab_algo/')
mlEngine.addpath(s, nargout = 0)


modelArr = ["PlainXGBoost", "FedXGBoost", "SecureBoost", "PseudoSecureBoost"]
dataset = ["Iris", "GiveMeCredits", "Adult", "DefaultCredits", "AugData"]

CONFIG = {
  "model": modelArr[1],
  "dataset": dataset[0],
}

class SIM_PARAM:
  N_SAMPLE = int(1e3)
  N_FEATURE = int(1500)

"""
Testing: nUsers
Dataset: GivemeCredits
N: 10k, 20k, 30k, 50k, 80k 120

"""
TEST_CASE = "NFEATURE_AugData"



np.random.seed(10)
N_CLIENTS = 5
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger = logging.getLogger()
day = date.today().strftime("%b-%d-%Y")

curTime = round(time.time())

logName = 'Log/Test_{}/{}/{}_{}_{}/Rank_{}.log'.format(TEST_CASE, str(day), str(curTime), str(CONFIG["dataset"]), str(CONFIG["model"]), str(rank))
os.makedirs(os.path.dirname(logName), exist_ok=True)

file_handler = logging.FileHandler(logName, mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
formatter = logging.Formatter('%(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

