from __future__ import division, print_function
import numpy as np

def fast_bin(X, a, b, N, weights=None, cyclic=False):
    """
    Fast binning.

    :note: cyclic parameter is ignored. Present only for compatibility with fast_linbin
    """
    Y = (X - a)
    delta = (b-a) / N
    Y /= delta
    iY = np.floor(Y).astype(int)
    return np.bincount(iY, weights=weights, minlength=N), np.linspace(a + delta/2, b - delta/2, N)

#def fast_linbin(X, a, b, N, weights = None, cyclic = False):
    #"""
    #Fast linear binning with added weighting
    #"""
    #X = np.atleast_1d(X).astype(float)
    #assert len(X.shape) == 1, "Error, X must be a 1D array"
    #if weights is not Nonw:
        #weights = np.atleast_1d(weights).astype(float)
        #assert weights.shape == X.shape, "Error, weights must be None or an array with the same shape as X"
    #delta = (b - a) / N
    #if cyclic:
        #lower = 0
        #upper = M
        #shift = -a
    #else:
        #lower = -0.5
        #upper = M-0.5
        #shift = -a-delta/2
    #Y = X + shift
    #Y /= delta
    #iY = np.floor(Y).astype(int)
    #rem = (Y - iY)
    #if weights is not None:
        #rem *= weights
    #if not cyclic:
        #iY += 1
    #c1 = np.bincount(iY, weights = 1-rem, minlength=N+1)
    #c2 = np.bincount(iY, weights = rem, minlength=N+1)
    #if cyclic:
        #c1
