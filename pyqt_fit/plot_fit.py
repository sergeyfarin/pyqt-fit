"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This modules implement functions to test and plot parametric regression.
"""

from __future__ import division, print_function, absolute_import
from numpy import argsort, std, abs, sqrt, arange, pi, c_, asarray
from pylab import figure, title, legend, plot, xlabel, ylabel, subplot, ylim, hist, suptitle, gca
from .compat import izip
from itertools import chain
from scipy.special import erfinv, gamma
from scipy import stats
#from .kernel_smoothing import LocalLinearKernel1D
from .nonparam_regression import NonParamRegression
from .compat import unicode_csv_writer as csv_writer
from collections import namedtuple

import sys
if sys.version_info >= (3,):
    CSV_WRITE_FLAGS = "wt"
else:
    CSV_WRITE_FLAGS = "wb"


def plot_dist_residuals(res):
    """
    Plot the distribution of the residuals.

    :returns: the handle toward the histogram and the plot of the fitted normal distribution
    """
    ph = hist(res, normed=True)
    xr = arange(res.min(), res.max(), (res.max() - res.min()) / 1024)
    yr = stats.norm(0, res.std()).pdf(xr)
    pn = plot(xr, yr, 'r--')
    xlabel('Residuals')
    ylabel('Frequency')
    title('Distributions of the residuals')
    return ph, pn


def plot_residuals(xname, xdata, res_desc, res):
    """
    Plot the residuals against the X axis

    :param str     xname:    Name of the X axis
    :param ndarray xdata:    1D array with the X data
    :param str     res_desc: Name of the Y axis
    :param ndarray res: 1D   array with the residuals

    The shapes of ``xdata`` and ``res`` must be the same

    :returns: The handles of the the plots of the residuals and of the smoothed residuals.
    """
    p_res = plot(xdata, res, '+', label='residuals')[0]
    plot([xdata.min(), xdata.max()], [0, 0], 'r--')
    av = NonParamRegression(xdata, res)
    av.fit()
    xr = arange(xdata.min(), xdata.max(), (xdata.max() - xdata.min()) / 1024)
    rr = av(xr)
    p_smooth = plot(xr, rr, 'g', label='smoothed residuals')
    xlabel(xname)
    ylabel("Residuals")
    ymin, ymax = ylim()
    ymax = max(ymax, -ymin)
    ylim(-ymax, ymax)
    title("Residuals (%s) vs. fitted" % (res_desc,))
    return p_res, p_smooth


def scaled_location_plot(yname, yopt, scaled_res):
    """
    Plot the scaled location, given the dependant values and scaled residuals.

    :param str     yname:      Name of the Y axis
    :param ndarray yopt:       Estimated values
    :param ndarray scaled_res: Scaled residuals

    :returns: the handles for the data and the smoothed curve
    """

    scr = sqrt(abs(scaled_res))
    p_scaled = plot(yopt, scr, '+')[0]
    av = NonParamRegression(yopt, scr)
    av.fit()
    xr = arange(yopt.min(), yopt.max(), (yopt.max() - yopt.min()) / 1024)
    rr = av(xr)
    p_smooth = plot(xr, rr, 'g')[0]
    expected_mean = 2 ** (1 / 4) * gamma(3 / 4) / sqrt(pi)
    plot([yopt.min(), yopt.max()], [expected_mean, expected_mean], 'r--')
    title('Scale-location')
    xlabel(yname)
    ylabel('$|$Normalized residuals$|^{1/2}$')
    gca().set_yticks([0, 1, 2])
    return [p_scaled, p_smooth]


def qqplot(scaled_res, normq):
    """
    Draw a Q-Q Plot from the sorted, scaled residuals (i.e. residuals sorted
    and normalized by their standard deviation)

    :param ndarray scaled_res: Scaled residuals
    :param ndarray normq:      Expected value for each scaled residual, based on its quantile.

    :returns: handle to the data plot
    """
    qqp = []
    qqp += plot(normq, scaled_res, '+')
    qqp += plot(normq, normq, 'r--')
    xlabel('Theoretical quantiles')
    ylabel('Normalized residuals')
    title('Normal Q-Q plot')
    return qqp

ResultStruct = namedtuple('ResultStruct', "fct fct_desc param_names xdata ydata xname yname "
                          "res_name residuals popt res yopts eval_points interpolation "
                          "sorted_yopts scaled_res normq CI CIs CIresults")


def fit_evaluation(fit, xdata, ydata, eval_points=None,
                   CI=(), CIresults = None, xname="X", yname="Y",
                   fct_desc=None, param_names=(), residuals=None, res_name='Standard'):
    """
    This function takes the output of a curve fitting experiment and store all the relevant
    information for evaluating its success in the result.

    :type  fit: fitting object
    :param fit: object configured for the fitting

    :type  xdata: ndarray of shape (N,) or (k,N) for function with k prefictors
    :param xdata: The independent variable where the data is measured

    :type  ydata: ndarray
    :param ydata: The dependant data

    :type  eval_points: ndarray or None
    :param eval_points: Contain the list of points on which the result must be expressed. It is
        used both for plotting and for the bootstrapping.

    :type  CI: tuple of int
    :param CI: List of confidence intervals to calculate. If empty, none are calculated.

    :type  xname: string
    :param xname: Name of the X axis

    :type  yname: string
    :param yname: Name of the Y axis

    :type  fct_desc: string
    :param fct_desc: Formula of the function

    :type  param_names: tuple of strings
    :param param_names: Name of the various parameters

    :type  residuals: callable
    :param residuals: Residual function

    :type  res_desc: string
    :param res_desc: Description of the residuals

    :rtype: :py:class:`ResultStruct`
    :returns: Data structure summarising the fitting and its evaluation
    """
    popt = fit.popt
    res = fit.res

    if CI:
        CIs = CIresults.CIs
    else:
        CIs = []

    yopts = fit(xdata)
    if eval_points is None:
        yvals = yopts
        eval_points = xdata
    else:
        yvals = fit(eval_points)

    scaled_res, res_IX, prob, normq = residual_measures(res)
    sorted_yopts = yopts[res_IX]

    result = {}
    result["fct"] = fit
    result["fct_desc"] = fct_desc
    result["param_names"] = param_names
    result["xdata"] = xdata
    result["ydata"] = ydata
    result["xname"] = xname
    result["yname"] = yname
    result["res_name"] = res_name
    result["residuals"] = residuals
    #result["args"] = fit.args
    result["popt"] = popt
    result["res"] = res
    result["yopts"] = yopts
    result["eval_points"] = eval_points
    result["interpolation"] = yvals
    result["sorted_yopts"] = sorted_yopts
    result["scaled_res"] = scaled_res
    result["normq"] = normq
    result["CI"] = CI
    result["CIs"] = CIs
    #result["CIparams"] = CIparams
    result["CIresults"] = CIresults
    #print("estimate jacobian = %s" % result["extra_output"][-1]["est_jacobian"])
    return ResultStruct(**result)

ResidualMeasures = namedtuple("ResidualMeasures", "scaled_res res_IX prob normq")


def residual_measures(res):
    """
    Compute quantities needed to evaluate the quality of the estimation, based solely
    on the residuals.

    :rtype: :py:class:`ResidualMeasures`
    :returns: the scaled residuals, their ordering, the theoretical quantile for each residuals,
        and the expected value for each quantile.
    """
    IX = argsort(res)
    scaled_res = res[IX] / std(res)

    prob = (arange(len(scaled_res)) + 0.5) / len(scaled_res)
    normq = sqrt(2) * erfinv(2 * prob - 1)

    return ResidualMeasures(scaled_res, IX, prob, normq)

_restestfields = "res_figure residuals scaled_residuals qqplot dist_residuals"
ResTestResult = namedtuple("ResTestResult", _restestfields)
Plot1dResult = namedtuple("Plot1dResult", "figure estimate data CIs " + _restestfields)


def plot1d(result, loc=0, fig=None, res_fig=None):
    """
    Use matplotlib to display the result of a fit, and return the list of plots used

    :rtype: :py:class:`Plot1dResult`
    :returns: hangles to the various figures and plots
    """
    if fig is None:
        fig = figure()
    else:
        try:
            figure(fig)
        except TypeError:
            figure(fig.number)

    p_est = plot(result.eval_points, result.interpolation, label='estimated')[0]
    p_data = plot(result.xdata, result.ydata, '+', label='data')[0]
    p_CIs = []
    if result.CI:
        for p, (low, high) in izip(result.CI, result.CIs[0]):
            l = plot(result.eval_points, low, '--', label='%g%% CI' % (p,))[0]
            h = plot(result.eval_points, high, l.get_color() + '--')[0]
            p_CIs += [l, h]
    if result.param_names:
        param_strs = ", ".join("%s=%g" % (n, v) for n, v in izip(result.param_names, result.popt))
    else:
        param_strs = ", ".join("%g" % v for v in result.popt)
    param_strs = "$%s$" % (param_strs,)

    title("Estimated function %s with params %s" % (result.fct_desc, param_strs))

    xlabel(result.xname)
    ylabel(result.yname)
    legend(loc=loc)

    plots = {"figure": fig, "estimate": p_est, "data": p_data, "CIs": p_CIs}

    prt = plot_residual_tests(result.xdata, result.yopts, result.res,
                              "{0} with params {1}".format(result.fct_desc, param_strs),
                              result.xname, result.yname, result.res_name, result.sorted_yopts,
                              result.scaled_res,
                              result.normq, res_fig)

    plots.update(prt._asdict())

    return Plot1dResult(**plots)


def plot_residual_tests(xdata, yopts, res, fct_name, xname="X", yname='Y', res_name="residuals",
                        sorted_yopts=None, scaled_res=None, normq=None, fig=None):
    """
    Plot, in a single figure, all four residuals evaluation plots: :py:func:`plot_residuals`,
    :py:func:`plot_dist_residuals`, :py:func:`scaled_location_plot` and :py:func:`qqplot`.

    :param ndarray xdata:        Explaining variables
    :param ndarray yopt:         Optimized explained variables
    :param str     fct_name:     Name of the fitted function
    :param str     xname:        Name of the explaining variables
    :param str     yname:        Name of the dependant variables
    :param str     res_name:     Name of the residuals
    :param ndarray sorted_yopts: ``yopt``, sorted to match the scaled residuals
    :param ndarray scaled_res:   Scaled residuals
    :param ndarray normq:        Estimated value of the quantiles for a normal distribution

    :type  fig: handle or None
    :param fig: Handle of the figure to put the plots in, or None to create a new figure

    :rtype: :py:class:`ResTestResult`
    :returns: The handles to all the plots
    """
    if fig is None:
        fig = figure()
    else:
        try:
            figure(fig)
        except TypeError:
            figure(fig.number)

    xdata = asarray(xdata)
    yopts = asarray(yopts)
    res = asarray(res)

    subplot(2, 2, 1)
# First subplot is the residuals
    if len(xdata.shape) == 1 or xdata.shape[1] == 1:
        p_res = plot_residuals(xname, xdata.squeeze(), res_name, res)
    else:
        p_res = plot_residuals(yname, yopts, res_name, res)

    if scaled_res is None or sorted_yopts is None or normq is None:
        scaled_res, res_IX, _, normq = residual_measures(res)
        sorted_yopts = yopts[res_IX]

    subplot(2, 2, 2)
    p_scaled = scaled_location_plot(yname, sorted_yopts, scaled_res)

    subplot(2, 2, 3)
# Q-Q plot
    qqp = qqplot(scaled_res, normq)

    subplot(2, 2, 4)
# Distribution of residuals
    drp = plot_dist_residuals(res)

    suptitle("Residual Test for {}".format(fct_name))

    return ResTestResult(fig, p_res, p_scaled, qqp, drp)


def write1d(outfile, result, res_desc, CImethod):
    """
    Write the result of a fitting and its evaluation to a CSV file.

    :param str          outfile:  Name of the file to write to
    :param ResultStruct result:   Result of the fitting evaluation
        (e.g. output of :py:func:`fit_evaluation`)
    :param str          res_desc: Description of the residuals
        (in more details than just the name of the residuals)
    :param str          CImethod: Description of the confidence interval estimation method
    """
    with open(outfile, CSV_WRITE_FLAGS) as f:
        w = csv_writer(f)
        w.writerow(["Function", result.fct.fct.description])
        w.writerow(["Residuals", result.res_name, res_desc])
        w.writerow(["Parameter", "Value"])
        for pn, pv in izip(result.param_names, result.popt):
            w.writerow([pn, "%.20g" % pv])
        #TODO w.writerow(["Regression Evaluation"])
        w.writerow([])
        w.writerow(["Data"])
        w.writerow([result.xname, result.yname, result.fct_desc, "Residuals: %s" % result.res_name])
        w.writerows(c_[result.xdata, result.ydata, result.yopts, result.res])
        w.writerow([])
        w.writerow(['Model validation'])
        w.writerow([result.yname, 'Normalized residuals', 'Theoretical quantiles'])
        w.writerows(c_[result.sorted_yopts, result.scaled_res, result.normq])
        if result.eval_points is not result.xdata:
            w.writerow([])
            w.writerow(["Interpolated data"])
            w.writerow([result.xname, result.yname])
            w.writerows(c_[result.eval_points, result.interpolation])
        if result.CI:
            w.writerow([])
            w.writerow(["Confidence interval"])
            w.writerow(["Method", CImethod])
            head = ["Parameters"] + \
                list(chain(*[["%g%% - low" % v, "%g%% - high" % v] for v in result.CI]))
            w.writerow(head)
            #print(result.CIs[1])
            for cis in izip(result.param_names, *chain(*result.CIs[1])):
                cistr = [cis[0]] + ["%.20g" % v for v in cis[1:]]
                w.writerow(cistr)
            w.writerow([result.yname])
            head[0] = result.xname
            w.writerow(head)
            w.writerows(c_[tuple(chain([result.eval_points], *result.CIs[0]))])


# /home/barbier/prog/python/curve_fitting/test.csv
