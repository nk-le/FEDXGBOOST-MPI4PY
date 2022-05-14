import numpy as np
from mpi4py import MPI
import logging
from datetime import date
import time 
import os

modelArr = ["PlainXGBoost", "FedXGBoost", "SecureBoost"]
dataset = ["Iris", "GiveMeCredits", "Adult", "DefaultCredits"]

CONFIG = {
  "model": modelArr[0],
  "dataset": dataset[2],
}

class SIM_PARAM:
  N_SAMPLE = None




np.random.seed(10)
N_CLIENTS = 5
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger = logging.getLogger()
day = date.today().strftime("%b-%d-%Y")

curTime = round(time.time())

logName = 'Log/Evaluation/{}/{}_{}_{}/Rank_{}.log'.format(str(day), str(curTime), str(CONFIG["dataset"]), str(CONFIG["model"]), str(rank))
os.makedirs(os.path.dirname(logName), exist_ok=True)

file_handler = logging.FileHandler(logName, mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
formatter = logging.Formatter('%(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

