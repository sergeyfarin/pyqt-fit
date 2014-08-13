from __future__ import division
from numpy import log, exp


class Standard(object):
    name = "Standard"
    description = "y_1 - y_0"

    @staticmethod
    def __call__(y1, y0):
        return y1 - y0

    @staticmethod
    def invert(y, res):
        """
        Add the residual to the value
        """
        return y + res

    @staticmethod
    def Dfun(y1, y0, dy):
        """
        J(y1-y0) = J(y1)-J(y0) = -J(y0)
        where J is the jacobian
        """
        return -1


class LogResiduals(object):
    name = "Difference of the logs"
    description = "log(y1/y0)"

    @staticmethod
    def __call__(y1, y0):
        return log(y1 / y0)

    @staticmethod
    def Dfun(y1, y0):
        """
        d(log(y1/y0))/dy0 = -1/y0
        where J is the jacobian and division is element-wise (per row)
        """
        return -1 / y0

    @staticmethod
    def invert(y, res):
        """
        Multiply the value by the exponential of the residual
        """
        return y * exp(res)
