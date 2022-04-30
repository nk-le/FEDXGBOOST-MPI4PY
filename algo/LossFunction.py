import numpy as np

class LeastSquareLoss:
    def gradient(self, actual, predicted):
        return -(actual - predicted)

    def hess(self, actual, predicted):
        return np.ones_like(actual)

class LogLoss():
    def gradient(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob - actual

    def hess(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob * (1.0 - prob) 