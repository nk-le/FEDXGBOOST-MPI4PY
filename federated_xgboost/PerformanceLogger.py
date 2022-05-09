import timeit

from common.Common import logger, rank
from numpy import append


def get_current_time():
    return timeit.default_timer()

class PerformanceLogger:

    def __init__(self) -> None:
        self.tStartBoosting = get_current_time()
        self.tEndBoosting = get_current_time()
        self.dtTree = []
        self.dtPred = []
        self.dtTotal = get_current_time()

    def log_start_boosting(self):
        self.tStartBoosting = get_current_time()
        return self.tStartBoosting

    def log_dt_tree(self, tStart: float):
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

    def print_info(self):
        logger.warning("Boosting performance\ndtBoost: %.1f| nTree = %d \ndtTree: %s\ndtPred: %s", 
                self.dtTotal, len(self.dtTree), str(self.dtTree), str(self.dtPred))