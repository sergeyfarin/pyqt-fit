from numpy import log, exp, newaxis

def standard(y1, y0):
    """
    Name: Standard
    Formula: y_1 - y_0
    Invert: add_standard
    Dfun: deriv_standard
    """
    return y1-y0

def add_standard(y, res):
    """
    Add the residual to the value
    """
    return y+res

def deriv_standard(y1, y0, dy):
    """
    J(y1-y0) = J(y1)-J(y0) = -J(y0)
    where J is the jacobian
    """
    return -dy

def log_residual(y1, y0):
    """
    Name: Difference of the logs
    Formula: log(y1/y0)
    Invert: add_log_residual
    Dfun: deriv_log
    """
    return log(y1/y0)

def deriv_log(y1, y0, dy):
    """
    J(log(y1/y0)) = -J(y0)/y0
    where J is the jacobian and division is element-wise (per row)
    """
    return -dy/y0[newaxis,:]

def add_log_residual(y, res):
    """
    Multiply the value by the exponential of the residual
    """
    return y*exp(res)
