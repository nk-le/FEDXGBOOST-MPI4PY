import numpy as np
from mpi4py import MPI
import logging
from datetime import date
import time 
import os 

test_dataset = "GiveMeSomeCredits"
loss_function = "LogLoss" 

modelArr = ["PlainXGBoost", "FedXGBoost", "SecureBoost"]
dataset = ["Iris", "GiveMeCredits", "Adult", "DefaultCredits"]

CONFIG = {
  "model": modelArr[2],
  "dataset": dataset[0],
}


np.random.seed(10)
N_CLIENTS = 5
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger = logging.getLogger()
day = date.today().strftime("%b-%d-%Y")

curTime = round(time.time())

logName = 'Log/{}/{}_{}_{}/Rank_{}.log'.format(str(day), str(curTime), str(CONFIG["dataset"]), str(CONFIG["model"]), str(rank))
os.makedirs(os.path.dirname(logName), exist_ok=True)

file_handler = logging.FileHandler(logName, mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
formatter = logging.Formatter('%(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

