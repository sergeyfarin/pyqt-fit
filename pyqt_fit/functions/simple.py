from numpy import exp, argsort, log, zeros, ones, array, log

def linear(xxx_todo_changeme, x):
    """
    Function: y = a x + b
    Parameters: a b
    Name: Linear
    ParametersEstimate: linearParams
    Dfun: deriv_linear
    """
    (a,b) = xxx_todo_changeme
    return a*x + b

def deriv_linear(xxx_todo_changeme1, x):
    (a,b) = xxx_todo_changeme1
    result = ones((2, x.shape[0]), dtype=x.dtype)
    result[0] = x # d/da
    # d/db = 1
    return result

def linearParams(x, y):
    b = y.min()
    a = (y.max() - y.min())/(x.max() - x.min())
    return (a,b)

def exponential(xxx_todo_changeme2, x):
    """
    Function: y = A e^{k(x-x_0)} + y_0
    Parameters: A k x_0 y_0
    Name: Exponential
    ParametersEstimate: exponentialParams
    Dfun: deriv_exponential
    """
    (A,k,x0,y0) = xxx_todo_changeme2
    return A*exp(k*(x-x0))+y0

def deriv_exponential(xxx_todo_changeme3, x):
    (A,k,x0,y0) = xxx_todo_changeme3
    result = ones((4, x.shape[0]), dtype=x.dtype)
    dx = x-x0
    ee = exp(k*dx)
    result[0] = ee # d/dA
    result[1] = dx*A*ee # d/dk
    result[2] = -A*ee # d/dx0
    # d/dy0 = 1
    return result

def exponentialParams(x, y):
    x0 = (x.max() + x.min())/2
    IX = argsort(x)
    xs = x[IX]
    ys = y[IX]
    k = log(ys[-1]/ys[0])/(xs[-1]-xs[0])
    A = ys[-1]/(exp(k*(xs[-1]-x0)))
    return (A,k,x0,ys[0])

def power_law(xxx_todo_changeme4, x):
    """
    Function: y = A (x-x_0)^k + y_0
    Name: Power law
    Parameters: A x_0 k y_0
    ParametersEstimate: power_lawParams
    Dfun: deriv_power_law
    """
    (A, x0, k, y0) = xxx_todo_changeme4
    return A*(x-x0)**k + y0

def deriv_power_law(xxx_todo_changeme5, x):
    (A, x0, k, y0) = xxx_todo_changeme5
    result = ones((4,x.shape[0]), dtype=x.dtype)
    dx = x-x0
    dxk1 = dx**(k-1)
    dxk = dxk1*dx
    result[0] = dxk # d/dA
    result[1] = -A*k*dxk1 # d/dx0
    result[2] = dxk*A*log(dx) # d/dk
    # d/dy0 = 1
    return result


def power_lawParams(x, y):
    x0 = x.min()
    y0 = y.min()
    A = (y.max()-y0)/(x.max()-x0)
    return (A, x0, 1, y0)

def logistic(xxx_todo_changeme6, x):
    """
    Function: y = A / (1 + e^{-k (x-x_0)}) + y_0
    Parameters: A k x_0 y_0
    Name: Logistic
    ParametersEstimate: logisticParams
    Dfun: deriv_logistic
    """
    (A, k, x0, y0) = xxx_todo_changeme6
    return A/(1+exp(k*(x0-x))) + y0

def deriv_logistic(xxx_todo_changeme7, x):
    (A, k, x0, y0) = xxx_todo_changeme7
    result = ones((4, x.shape[0]), dtype=x.dtype)
    dx = x0-x
    ee = exp(k*dx)
    ee1 = ee+1.
    ee2 = ee1*ee1
    result[0] = 1./ee1 # d/dA
    result[1] = -dx*A*ee/ee2 # d/dk
    result[2] = -A*k*ee/ee2 # d/dx0
    # d/dy0 = 1
    return result

def logisticParams(x, y):
    x0 = (x.max()+x.min())/2
    k = log(y.min())/(x.min()-x0)
    return (y.max(), k, x0, y.min())

