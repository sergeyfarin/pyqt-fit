from numpy import exp, argsort, log

def linear(x, (a,b)):
    """
    Function: y = a x + b
    Parameters: a b
    Name: Linear
    ParametersEstimate: linearParams
    """
    return a*x + b

def linearParams(x, y):
    b = y.min()
    a = (y.max() - y.min())/(x.max() - x.min())
    return (a,b)

def exponential(x, (A,k,x0,y0)):
    """
    Function: y = A e^{k(x-x_0)} + y_0
    Parameters: A k x_0 y_0
    Name: Exponential
    ParametersEstimate: exponentialParams
    """
    return A*exp(k*(x-x0))+y0

def exponentialParams(x, y):
    x0 = (x.max() + x.min())/2
    IX = argsort(x)
    xs = x[IX]
    ys = y[IX]
    k = log(ys[-1]/ys[0])/(xs[-1]-xs[0])
    A = ys[-1]/(exp(k*(xs[-1]-x0)))
    return (A,k,x0,ys[0])

def power_law(x, (A, x0, k, y0)):
    """
    Function: y = A (x-x_0)^k + y_0
    Name: Power law
    Parameters: A x_0 k y_0
    ParametersEstimate: power_lawParams
    """
    return A*(x-x0)**k + y0

def power_lawParams(x, y):
    x0 = x.min()
    y0 = y.min()
    A = (y.max()-y0)/(x.max()-x0)
    return (A, x0, 1, y0)

def logistic(x, (A, k, x0, y0)):
    """
    Function: y = A / (1 + e^{-k (x-x_0)}) + y_0
    Parameters: A k x_0 y_0
    Name: Logistic
    ParametersEstimate: logisticParams
    """
    return A/(1+exp(k*(x0-x))) + y0

def logisticParams(x, y):
    x0 = (x.max()+x.min())/2
    k = log(y.min())/(x.min()-x0)
    return (y.max(), k, x0, y.min())

