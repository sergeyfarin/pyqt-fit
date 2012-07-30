from numpy import log, exp

def standard(y1, y0):
    """
    Name: Standard
    Formula: y_1 - y_0
    Invert: add_standard
    """
    return y1-y0

def add_standard(y, res):
    """
    Add the residual to the value
    """
    return y+res

def log_residual(y1, y0):
    """
    Name: Difference of the logs
    Formula: log(y1/y0)
    Invert: add_log_residual
    """
    return log(y1/y0)

def add_log_residual(y, res):
    """
    Multiply the value by the exponential of the residual
    """
    return y*exp(res)
