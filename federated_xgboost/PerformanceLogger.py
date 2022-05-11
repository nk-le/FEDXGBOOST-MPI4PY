import timeit

from config import logger, rank


def get_current_time():
    return timeit.default_timer()

class CommunicationLogger:
    def __init__(self) -> None:
        self.tx = []
        self.rx = []

    def log_nRx(self, nRx):
        self.rx.append(nRx)

    def log_nTx(self, nTx):
        self.tx.append(nTx)

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

    def log_dt_fit(self, tStart: float):
        dt = get_current_time() - tStart
        self.dtTree.append(dt)

    def log_dt_pred(self, tStart: float):
        dt = get_current_time() - tStart 
        self.dtPred.append(dt)        

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