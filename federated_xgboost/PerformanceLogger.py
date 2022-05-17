import timeit

from config import logger, rank


def get_current_time():
    return timeit.default_timer()

class CommunicationLogger:
    def __init__(self, nClients) -> None:
        self.nClients = nClients
        self.tx = [[] for i in range(self.nClients)]
        self.rx = [[] for i in range(self.nClients)]

    def log_nRx(self, nRx, i = 0, treeID = 0):
        self.rx[i].append(nRx)
        logger.warning("CommunicationRX, TreeID: %d, nRx: %s, Partner: %d", treeID, nRx, i)

    def log_nTx(self, nTx, i = 0, treeID = 0):
        self.tx[i].append(nTx)
        logger.warning("CommunicationTX, TreeID: %d, nTx: %s, Partner: %d", treeID, nTx, i)

    def log(self):
        logger.warning("Communication, nRx: %s, nTx: %s", str(self.rx), str(self.tx))


class TimeLogger:
    def __init__(self) -> None:
        self.tStartBoosting = get_current_time()
        self.tEndBoosting = get_current_time()
        self.dtTree = []
        self.dtPred = []
        self.dtTotal = get_current_time()

    def log_start_boosting(self):
        self.tStartBoosting = get_current_time()
        return self.tStartBoosting

    def log_dt_fit(self, tStart: float, treeID = 0):
        dt = get_current_time() - tStart
        self.dtTree.append(dt)

        logger.warning("FitTime, TreeID: %d,  dt: %s", treeID, dt)

    def log_dt_pred(self, tStart: float, treeID = 0):
        dt = get_current_time() - tStart 
        self.dtPred.append(dt)
        logger.warning("PredTime, TreeID: %d,  dt: %s", treeID, dt)        

    def tic():
        return get_current_time()

    def toc(tic):
        return get_current_time() - tic

    def log_end_boosting(self, tStart):
        self.tEndBoosting = get_current_time()
        self.dtTotal = get_current_time() - tStart
        

    def log(self):
        logger.warning("ExecutionTime, dtBoost: %.1f, nTree = %d, dtTree: %s, dtPred: %s", 
                self.dtTotal, len(self.dtTree), str(self.dtTree), str(self.dtPred))