"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This modules provides function for bootstrapping a regression method.
"""

from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.random import randint
from scipy import optimize
from collections import namedtuple
from . import nonparam_regression
from . import sharedmem
import multiprocessing as mp
from . import bootstrap_workers
from .compat import irange, izip


def percentile(array, p, axis=0):
    """
    Compute the percentiles of the values in array
    """
    a = np.asarray(array).sort(axis=axis)
    n = (len(a) - 1) * p / 100
    n0 = np.floor(n)
    n1 = n0 + 1
    #print("%g percentile on %d = [%d-%d]" % (p*100, len(array), n0, n1))
    d = n - n0
    v0 = array[n0]
    v1 = array[n1]
    return v0 + d * (v1 - v0)


def bootstrap_residuals(fct, xdata, ydata, repeats=3000, residuals=None,
                        add_residual=None, correct_bias=False, **kwrds):
    """
    This implements the residual bootstrapping method for non-linear
    regression.

    :type  fct: callable
    :param fct: Function evaluating the function on xdata at least with
        ``fct(xdata)``

    :type  xdata: ndarray of shape (N,) or (k,N) for function with k predictors
    :param xdata: The independent variable where the data is measured

    :type  ydata: ndarray
    :param ydata: The dependant data

    :type  residuals: ndarray or callable or None
    :param residuals: Residuals for the estimation on each xdata. If callable,
        the call will be ``residuals(ydata, yopt)``.

    :type  repeats: int
    :param repeats: Number of repeats for the bootstrapping

    :type  add_residual: callable or None
    :param add_residual: Function that add a residual to a value. The call
        ``add_residual(yopt, residual)`` should return the new ydata, with
        the residuals 'applied'. If None, it is considered the residuals should
        simply be added.

    :type  correct_bias: boolean
    :param correct_bias: If true, the additive bias of the residuals is
        computed and restored

    :type  kwrds: dict
    :param kwrds: Dictionnary present to absorbed unknown named parameters

    :rtype: (ndarray, ndarray)
    :returns:

        1. xdata, with a new axis at position -2. This correspond to the
        'shuffled' xdata (as they are *not* shuffled here)

        2.Second item is the shuffled ydata. There is a line per repeat, each
        line is shuffled independently.

    .. todo::

        explain the method here, as well as how to create add_residual
    """
    if residuals is None:
        residuals = np.subtract

    yopt = fct(xdata)

    if not isinstance(residuals, np.ndarray):
        res = residuals(ydata, yopt)
    else:
        res = np.array(residuals)

    res -= np.mean(res)

    shuffle = randint(0, len(ydata), size=(repeats, len(ydata)))

    shuffled_res = res[shuffle]

    if correct_bias:
        kde = nonparam_regression.NonParamRegression(xdata, res)
        kde.fit()
        bias = kde(xdata)
        shuffled_res += bias

    if add_residual is None:
        add_residual = np.add

    modified_ydata = add_residual(yopt, shuffled_res)

    return xdata[..., np.newaxis, :], modified_ydata


def bootstrap_regression(fct, xdata, ydata, repeats=3000, **kwrds):
    """
    This implements the shuffling of standard bootstrapping method for
    non-linear regression.

    :type  fct: callable
    :param fct: This is the function to optimize

    :type  xdata: ndarray of shape (N,) or (k,N) for function with k predictors
    :param xdata: The independent variable where the data is measured

    :type  ydata: ndarray
    :param ydata: The dependant data

    :type  repeats: int
    :param repeats: Number of repeats for the bootstrapping

    :type  kwrds: dict
    :param kwrds: Dictionnary to absorbed unknown named parameters

    :rtype: (ndarray, ndarray)
    :returns:
        1. The shuffled x data. The axis -2 has one element per repeat, the
        other axis are shuffled independently.

        2. The shuffled ydata. There is a line per repeat, each line is
        shuffled independently.

    .. todo::

        explain the method here
    """
    shuffle = randint(0, len(ydata), size=(repeats, len(ydata)))
    shuffled_x = xdata[..., shuffle]
    shuffled_y = ydata[shuffle]
    return shuffled_x, shuffled_y


def getCIs(CI, *arrays):
    #sorted_arrays = [ np.sort(a, axis=0) for a in arrays ]

    if not np.iterable(CI):
        CI = (CI,)

    def make_CI(a):
        return np.zeros((len(CI), 2) + a.shape[1:], dtype=float)
    CIs = tuple(make_CI(a) for a in arrays)
    for i, ci in enumerate(CI):
        ci = (100. - ci) / 2
        for cis, arr in izip(CIs, arrays):
            low = np.percentile(arr, ci, axis=0)
            high = np.percentile(arr, 100 - ci, axis=0)
            cis[i] = [low, high]

    return CIs

BootstrapResult = namedtuple('BootstrapResult', '''y_fit y_est eval_points y_eval CIs_val CIs
shuffled_xs shuffled_ys full_results''')


def bootstrap(fit, xdata, ydata, CI, shuffle_method=bootstrap_residuals,
              shuffle_args=(), shuffle_kwrds={}, repeats=3000,
              eval_points=None, full_results=False, nb_workers=None,
              extra_attrs=(), fit_args=(), fit_kwrds={}):
    """
    This function implement the bootstrap algorithm for a regression algorithm.
    It is capable of spreading the load across many threads using shared memory
    and the :py:mod:`multiprocess` module.

    :type  fit: callable
    :param fit:
        Method used to compute regression. The call is::

            f = fit(xdata, ydata, *fit_args, **fit_kwrds)

        Fit should return an object that would evaluate the regression on a
        set of points. The next call will be::

            f(eval_points)

    :type  xdata: ndarray of shape (N,) or (k,N) for function with k predictors
    :param xdata: The independent variable where the data is measured

    :type  ydata: ndarray
    :param ydata: The dependant data

    :type  CI: tuple of float
    :param CI: List of percentiles to extract

    :type  shuffle_method: callable
    :param shuffle_method:
        Create shuffled dataset. The call is::

          shuffle_method(xdata, ydata, y_est, repeat=repeats, *shuffle_args,
                         **shuffle_kwrds)

        where ``y_est`` is the estimated dependant variable on the xdata.

    :type  shuffle_args: tuple
    :param shuffle_args: List of arguments for the shuffle method

    :type  shuffle_kwrds: dict
    :param shuffle_kwrds: Dictionnary of arguments for the shuffle method

    :type  repeats: int
    :param repeats: Number of repeats for the bootstraping

    :type  eval_points: ndarray or None
    :param eval_points: List of points to evaluate. If None, eval_point
        is xdata.

    :type  full_results: bool
    :param full_results: if True, output also the whole set of evaluations

    :type  nb_workers: int or None
    :param nb_worders: Number of worker threads. If None, the number of
        detected CPUs will be used. And if 1 or less, a single thread
        will be used.

    :type  extra_attrs: tuple of str
    :param extra_attrs: List of attributes of the fitting method to extract on
        top of the y values for confidence intervals

    :type  fit_args: tuple
    :param fit_args: List of extra arguments for the fit callable

    :type  fit_kwrds: dict
    :param fit_kwrds: Dictionnary of extra named arguments for the fit callable

    :rtype: :py:class:`BootstrapResult`
    :return: Estimated y on the data, on the evaluation points, the requested
        confidence intervals and, if requested, the shuffled X, Y and the full
        estimated distributions.
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    y_fit = fit(xdata, ydata, *fit_args, **fit_kwrds)
    y_fit.fit()

    shuffled_x, shuffled_y = shuffle_method(y_fit, xdata, ydata,
                                            repeats=repeats,
                                            *shuffle_args, **shuffle_kwrds)
    nx = shuffled_x.shape[-2]
    ny = shuffled_y.shape[0]
    extra_values = []
    for attr in extra_attrs:
        extra_values.append(getattr(y_fit, attr))

    if eval_points is None:
        eval_points = xdata
    if nb_workers is None:
        nb_workers = mp.cpu_count()

    multiprocess = nb_workers > 1

# Copy everything in shared mem
    if multiprocess:
        ra = sharedmem.zeros((repeats + 1, len(eval_points)), dtype=float)
        result_array = ra.np
        sx = sharedmem.array(shuffled_x)
        sy = sharedmem.array(shuffled_y)
        ep = sharedmem.array(eval_points)

        def make_ea(ev):
            return sharedmem.zeros((repeats + 1, len(ev)), dtype=float)
        eas = [make_ea(ev) for ev in extra_values]
        extra_arrays = [ea.np for ea in eas]
        pool = mp.Pool(mp.cpu_count(), bootstrap_workers.initialize_shared,
                       (nx, ny, ra, eas, sx, sy, ep, extra_attrs,
                        fit, fit_args, fit_kwrds))
    else:
        result_array = np.empty((repeats + 1, len(eval_points)), dtype=float)

        def make_ea(ev):
            return np.empty((repeats + 1, len(ev)), dtype=float)
        extra_arrays = [make_ea(ev) for ev in extra_values]
        bootstrap_workers.initialize(nx, ny, result_array, extra_arrays,
                                     shuffled_x, shuffled_y, eval_points,
                                     extra_attrs, fit, fit_args, fit_kwrds)

    result_array[0] = y_fit(eval_points)

    for ea, ev in izip(extra_arrays, extra_values):
        ea[0] = ev

    base_repeat = repeats // nb_workers
    if base_repeat * nb_workers < repeats:
        base_repeat += 1

    for i in irange(nb_workers):
        end_repeats = (i + 1) * base_repeat
        if end_repeats > repeats:
            end_repeats = repeats
        if multiprocess:
            pool.apply_async(bootstrap_workers.bootstrap_result,
                             (i, i * base_repeat, end_repeats))
        else:
            bootstrap_workers.bootstrap_result(i, i * base_repeat, end_repeats)

    if multiprocess:
        pool.close()
        pool.join()
    CIs = getCIs(CI, result_array, *extra_arrays)

    # copy the array to not return a view on a larger array
    y_eval = np.array(result_array[0])

    if not full_results:
        shuffled_y = shuffled_x = result_array = None
        extra_arrays = ()
    elif multiprocess:
        result_array = result_array.copy()  # copy in local memory
        extra_arrays = [ea.copy for ea in extra_arrays]

    return BootstrapResult(y_fit, y_fit(xdata), eval_points, y_eval, tuple(CI), CIs,
                           shuffled_x, shuffled_y, result_array)


def test():
    import quad
    from numpy.random import rand, randn
    from pylab import plot, clf, legend, arange, figure, title, show
    from curve_fitting import curve_fit

    def quadratic(x, params):
        p0, p1, p2 = params
        return p0 + p1 * x + p2 * x ** 2
    #test = quadratic
    test = quad.quadratic

    init = (10, 1, 1)
    target = np.array([10, 4, 1.2])
    print("Target parameters: {}".format(target))
    x = 6 * rand(200) - 3
    y = test(x, target) * (1 + 0.3 * randn(x.shape[0]))
    xr = arange(-3, 3, 0.01)
    yr = test(xr, target)

    print("Estimage best parameters, fixing the first one")
    popt, pcov, _, _ = curve_fit(test, x, y, init, fix_params=(0,))
    print("Best parameters: {}".format(popt))

    print("Estimate best parameters from data")
    popt, pcov, _, _ = curve_fit(test, x, y, init)
    print("Best parameters: {}".format(popt))

    figure(1)
    clf()
    plot(x, y, '+', label='data')
    plot(xr, yr, 'r', label='function')
    legend(loc='upper left')

    print("Residual bootstrap calculation")
    result_r = bootstrap(test, x, y, init, (95, 99),
                         shuffle_method=bootstrap_residuals, eval_points=xr,
                         fit=curve_fit)
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

    print("Regression bootstrap calculation")
    (popt_c, pcov_c, res_c, CI_c, CIp_r,
     extra_c) = bootstrap(test, x, y, init, CI=(95, 99),
                          shuffle_method=bootstrap_regression, eval_points=xr,
                          fit=curve_fit)
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

    print("Done")

    show()

    return locals()


def profile(filename='bootstrap_profile'):
    import cProfile
    import pstats
    cProfile.run('res = bootstrap.test()', 'bootstrap_profile')
    p = pstats.Stats('bootstrap_profile')
    return p

if __name__ == "__main__":
    test()
