from numpy import iterable, zeros, floor, exp, sort, newaxis, array
from numpy.random import rand, randn, randint
from scipy import optimize

def adapt_curve_fit(fct, x, y, p0, args=(), **kwrds):
    popt, pcov = optimize.curve_fit(fct, x, y, **kwrds)
    return (popt, pcov, fct(x, popt, *args) - y)

def _percentile(array, p):
    n = len(array)*p
    n0 = floor(n)
    n1 = n0+1
    #print "%g percentile on %d = [%d-%d]" % (p*100, len(array), n0, n1)
    d = n-n0
    v0 = array[n0]
    v1 = array[n1]
    return v0 + d*(v1-v0)

def bootstrap_residuals(fct, xdata, ydata, popt, res, repeats = 3000, args = (), add_residual = None, **kwrds):
    """
    This implements the residual bootstrapping method for non-linear regression.

    Parameters
    ----------
    fct: callable
        This is the function to optimize
    xdata: ndarray of shape (N,) or (k,N) for function with k perdictors
        The independent variable where the data is measured
    ydata: ndarray
        The dependant data
    popt: ndarray
        Array of optimal parameters
    res: ndarray
        List of residuals for the given parameters
    repeats: int
        Number of repeats for the bootstrapping
    args: tuple
        Extra arguments for the ``fct`` function
    add_residual: callable or None
        Function that add a residual to a value. The call ``add_residual(ydata,
        residual)`` should return the new ydata, with the residuals 'applied'. If
        None, it is considered the residuals should simply be added.
    kwrds: dict
        Dictionnary to absorbed unknown named parameters

    Returns
    -------
    shuffled_x: ndarray
        Return xdata, with a new axis at position -2.
    shuffled_y: ndarray
        Return the shuffled ydata. There is a line per repeat, each line is shuffled independently.

    Notes
    -----
    TODO explain the method here, as well as how to create add_residual
    """

    shuffle = randint(0, len(ydata), size=(repeats, len(ydata)))
    shuffled_res = res[shuffle]

    if add_residual is None:
        add_residual = lambda y,r: y+r

    yopt = fct(xdata, popt, *args)
    modified_ydata = add_residual(yopt,shuffled_res)

    return xdata[...,newaxis,:], modified_ydata

def bootstrap_regression(fct, xdata, ydata, popt, res, repeats = 3000, eval_points = None, args = (), **kwrds):
    """
    This implements the shuffling of standard bootstrapping method for non-linear regression.

    Parameters
    ----------
    fct: callable
        This is the function to optimize
    xdata: ndarray of shape (N,) or (k,N) for function with k perdictors
        The independent variable where the data is measured
    ydata: ndarray
        The dependant data
    popt: ndarray
        Array of optimal parameters
    res: ndarray
        List of residuals for the given parameters
    repeats: int
        Number of repeats for the bootstrapping
    args: tuple
        Extra arguments for the ``fct`` function
    kwrds: dict
        Dictionnary to absorbed unknown named parameters

    Returns
    -------
    shuffled_x: ndarray
        Return the shuffled x data. The axis -2 has one element per repeat, the other axis are shuffled independently.
    shuffled_y: ndarray
        Return the shuffled ydata. There is a line per repeat, each line is shuffled independently.

    Notes
    -----
    TODO explain the method here
    """
    shuffle = randint(0, len(ydata), size=(repeats, len(ydata)))
    shuffled_x = xdata[...,shuffle]
    shuffled_y = ydata[shuffle]
    return shuffled_x, shuffled_y

def bootstrap(fct, xdata, ydata, p0, CI, shuffle_method = bootstrap_residuals, shuffle_args={}, repeats = 3000, eval_points = None, args=(), fit=adapt_curve_fit, fit_args={}):
    """
    Implement the standard bootstrap method applied to a regression method.

    Parameters
    ----------
    fct: callable
        Function calculating the output, given x, the call is ``fct(xdata, p0, *args)``
    xdata: ndarray of shape (N,) or (k,N) for function with k perdictors
        The independent variable where the data is measured
    ydata: ndarray of dimension (N,)
        The dependant data
    p0: ndarray
        Initial values for the estimated parameters
    CI: tuple of float
        List of percentiles to calculate
    shuffle_method: callable
        Create shuffled dataset. The call is:
        ``shuffle_method(fct, xdata, ydata, popt, residuals, repeat=repeats, args=args, **shuffle_args)``
    shuffle_args: dict
        Dictionnary of arguments for the shuffle method
    repeats: int
        Number of repeats for the bootstrapping
    eval_points: None or ndarray
        Point on which the function is evaluated for the output of the
        bootstrapping. If None, it will use xdata.
    args: tuple
        Extra arguments for the function ``fct``
    fit: callable
        Function used to estimate a new set of parameters. The call must be
        ``fit(fct, xdata, ydata, p0, args=args, **fit_args)`` and the
        first returned arguments are the estimated p, their covariance matrix
        and the residuals
    fit_args: dict
        Extra keyword arguments for the fit function

    Returns
    -------
    popt: ndarray
        Optimized parameters
    pcov : 2d array
        The estimated covariance of popt.  The diagonals provide the variance
        of the parameter estimate.
    res: ndarray
        Residuals for the optimal values
    CIs: list of pair of ndarray
        For each CI value, a pair of ndarray is provided for the lower and
        upper bound of the function on the points specified in eval_points
    CIparams: list of pair of ndarray
        For each CI value, a pair of ndarray is provided for the lower and
        upper bound of the parameters
    extra_output: tuple
        Any extra output of fit during the first evaluation

    Notes
    -----
    TODO explain the method here
    """
    result = fit(fct, xdata, ydata, p0, args=args, **fit_args)
    popt, pcov, residuals = result[:3]
    extra_output = result[3:]

    shuffled_x, shuffled_y = shuffle_method(fct, xdata, ydata, popt, residuals, repeats=repeats, args=args, **shuffle_args)
    nx = shuffled_x.shape[-2]
    ny = shuffled_y.shape[0]

    if eval_points is None:
        eval_points = xdata

    result_array = zeros((repeats+1, len(eval_points)), dtype=float)
    params_array = zeros((repeats+1, len(popt)), dtype=float)

    result_array[0] = fct(eval_points, popt, *args)
    params_array[0] = popt
    for i in xrange(0,repeats):
        new_result = fit(fct, shuffled_x[...,i%nx,:], shuffled_y[i%ny,:], popt, args=args, **fit_args)
        result_array[i+1] = fct(eval_points, new_result[0], *args)
        params_array[i+1] = new_result[0]

    CIs, CIparams = getCIs(CI, result_array, params_array)
    return popt, pcov, residuals, CIs, CIparams, extra_output

def getCIs(CI, result_array, params_array):
    sorted_array = sort(result_array, axis=0)
    sorted_params = sort(params_array, axis=0)

    if not iterable(CI):
        CI = (CI,)

    CIs = []
    CIparams = []
    for ci in CI:
        ci = (1-ci/100.0)/2
        low = _percentile(sorted_array, ci)
        high = _percentile(sorted_array, 1-ci)
        CIs.append((low, high))
        low = _percentile(sorted_params, ci)
        high = _percentile(sorted_params, 1-ci)
        CIparams.append((low, high))

    return CIs, CIparams

def test():
    import pyximport
    pyximport.install()
    import quad
    from numpy.random import rand, randn
    from pylab import plot, savefig, clf, legend, arange, figure, title, show
    from curve_fit import curve_fit
    import residuals

    def quadratic(x,(p0,p1,p2)):
        return p0 + p1*x + p2*x**2
    #test = quadratic
    test = quad.quadratic

    init = (10,1,1)
    target = array([10,4,1.2])
    print "Target parameters: %s" % (target,)
    x = 6*rand(200) - 3
    y = test(x, target)*(1+0.3*randn(x.shape[0]))
    xr = arange(-3, 3, 0.01)
    yr = test(xr,target)

    print "Estimage best parameters, fixing the first one"
    popt, pcov, _, _ = curve_fit(test, x, y, init, fix_params=(0,))
    print "Best parameters: %s" % (popt,)

    print "Estimate best parameters from data"
    popt, pcov, _, _ = curve_fit(test, x, y, init)
    print "Best parameters: %s" % (popt,)

    figure(1)
    clf()
    plot(x, y, '+', label='data')
    plot(xr, yr, 'r', label='function')
    legend(loc='upper left')

    print "Residual bootstrap calculation"
    result_r = bootstrap(test, x, y, init, (95, 99), shuffle_method=bootstrap_residuals, eval_points = xr, fit=curve_fit)
    popt_r, pcov_r, res_r, CI_r, CIp_r, extra_r = result_r
    yopt_r = test(xr, popt_r)

    figure(2)
    clf()
    plot(xr, yopt_r, 'g', label='estimate')
    plot(xr, yr, 'r', label='target')
    plot(xr, CI_r[0][0], 'b--', label='95% CI')
    plot(xr, CI_r[0][1], 'b--')
    plot(xr, CI_r[1][0], 'k--', label='99% CI')
    plot(xr, CI_r[1][1], 'k--')
    legend(loc='upper left')
    title('Residual Bootstrapping')

    print "Regression bootstrap calculation"
    popt_c, pcov_c, res_c, CI_c, CIp_r, extra_c = bootstrap(test, x, y, init, CI=(95, 99), shuffle_method=bootstrap_regression, eval_points = xr, fit=curve_fit)
    yopt_c = test(xr, popt_c)

    figure(3)
    clf()
    plot(xr, yopt_c, 'g', label='estimate')
    plot(xr, yr, 'r', label='target')
    plot(xr, CI_c[0][0], 'b--', label='95% CI')
    plot(xr, CI_c[0][1], 'b--')
    plot(xr, CI_c[1][0], 'k--', label='99% CI')
    plot(xr, CI_c[1][1], 'k--')
    legend(loc='upper left')
    title('Regression Bootstrapping (also called Case Resampling)')

    print "Done"

    show()

    return locals()

def profile(filename = 'bootstrap_profile'):
    import cProfile
    import pstats
    cProfile.run('res = bootstrap.test()', 'bootstrap_profile')
    p = pstats.Stats('bootstrap_profile')
    return p

if __name__ == "__main__":
    test()

