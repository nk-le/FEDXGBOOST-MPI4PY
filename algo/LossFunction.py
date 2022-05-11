import numpy as np

class LeastSquareLoss:
    def diff(self, actual, predicted):
        return sum((actual - predicted)**2)

    def gradient(self, actual, predicted):
        return -(actual - predicted)

    def hess(self, actual, predicted):
        return np.ones_like(actual)

class LogLoss():
    def diff(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return sum(actual * np.log(prob) + (1 - actual) * np.log(1 - prob))

    def gradient(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob - actual

    def hess(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob * (1.0 - prob) 