from __future__ import division
from curve_fitting import CurveFitting
from numpy import sort, iterable, argsort, std, abs, sqrt, arange, pi, c_
from pylab import figure, title, legend, plot, xlabel, ylabel, subplot, clf, ylim, hist, suptitle, gca
import bootstrap
from itertools import izip, chain
from scipy.special import erfinv, gamma
from scipy import stats
#try:
#    from cy_kernel_smoothing import SpatialAverage
#except ImportError:
from kernel_smoothing import SpatialAverage, LocalLinearKernel1D
import inspect
from csv import writer as csv_writer
from collections import namedtuple

smoothing = LocalLinearKernel1D

import sys
if sys.version_info >= (3,):
    CSV_WRITE_FLAGS = "wt"
else:
    CSV_WRITE_FLAGS = "wb"

def plot_dist_residuals(res):
    hist(res,normed=True)
    xr = arange(res.min(), res.max(), (res.max()-res.min())/1024)
    yr = stats.norm(0, res.std()).pdf(xr)
    plot(xr, yr, 'r--')
    xlabel('Residuals')
    ylabel('Frequency')
    title('Distributions of the residuals')

def plot_residuals(xname, xdata, res_desc, res):
    p_res = plot(xdata, res, '+', label='residuals')[0]
    plot([xdata.min(), xdata.max()], [0,0], 'r--')
    av = LocalLinearKernel1D(xdata, res)
    xr = arange(xdata.min(), xdata.max(), (xdata.max()-xdata.min())/1024)
    rr = av(xr)
    p_smooth = plot(xr, rr, 'g', label='smoothed residuals')
    xlabel(xname)
    ylabel("Residuals")
    ymin, ymax = ylim()
    ymax = max(ymax, -ymin)
    ylim(-ymax, ymax)
    title("Residuals (%s) vs. fitted" % (res_desc,))
    return p_res, p_smooth

def scaled_location_plot(yname, sorted_yopt, scaled_res):
    """
    Plot the scaled location, given the X and scaled residuals
    """
    scr = sqrt(abs(scaled_res))
    p_scaled = plot(sorted_yopt, scr, '+')[0]
    av = LocalLinearKernel1D(sorted_yopt, scr)
    xr = arange(sorted_yopt.min(), sorted_yopt.max(), (sorted_yopt.max() - sorted_yopt.min())/1024)
    rr = av(xr)
    p_smooth = plot(xr, rr, 'g')[0]
    expected_mean = 2**(1/4)*gamma(3/4)/sqrt(pi)
    plot([sorted_yopt.min(), sorted_yopt.max()], [expected_mean, expected_mean], 'r--')
    title('Scale-location')
    xlabel(yname)
    ylabel('$|$Normalized residuals$|^{1/2}$')
    gca().set_yticks([0,1,2])
    return [p_scaled, p_smooth]

def qqplot(scaled_res, normq):
    """
    Draw a Q-Q Plot from the sorted, scaled residuals (i.e. residuals sorted
    and normalized by their standard deviation)
    """
    qqp = []
    qqp += plot(normq, scaled_res, '+');
    qqp += plot(normq, normq, 'r--');
    xlabel('Theoretical quantiles');
    ylabel('Normalized residuals');
    title('Normal Q-Q plot');
    return qqp

ResultStruct = namedtuple('ResultStruct', "fct fct_desc param_names xdata ydata xname yname res_name residuals popt res yopts eval_points interpolation sorted_yopt scaled_res normq residuals_evaluation CI CIs CIresults")

def fit_evaluation(fit, xdata, ydata, eval_points=None,
        CI=(), CIresults = None, xname="X", yname="Y", fct_desc = None, param_names = (), residuals=None, res_name = 'Standard'):
    """
    Parameters
    ----------
    fit: fitting object
        object configured for the fitting
    xdata: ndarray of shape (N,) or (k,N) for function with k prefictors
        The independent variable where the data is measured
    ydata: ndarray
        The dependant data
    eval_points: ndarray or None
        Contain the list of points on which the result must be expressed. It is
        used both for plotting and for the bootstrapping.
    CI: tuple of int
        List of confidence intervals to calculate. If empty, none are calculated.
    xname: string
        Name of the X axis
    yname: string
        Name of the Y axis
    fct_desc: string
        Formula of the function
    param_names: tuple of strings
        Name of the various parameters
    residuals: callable
        Residual function
    res_desc: string
        Description of the residuals

    Returns
    -------
    This function returns an object with the following attributes:
    popt: ndarray
        Optimal values for the parameters as returns by the ``fit`` function
    res: ndarray
        Residuals for the optimal values
    yopts: ndarray
        Evaluation of the function with popt on xdata
    interpolated data: tuple
        eval_points: ndarray
            Values on which the function is evaluated
        yvals: ndarray
            Values of the function on these points
    residuals_evaluation: tuple
        X_sorted: ndarray
            x values sorted from the smallest to largest residual
        scaled_res: ndarray
            residuals sorted and scaled to be of variance 1
        normq: ndarray
            normalized quantile for the residuals
    CIs: list of pairs of array
        For each element of the CI argument, return a pair of array: the lower
        and upper bounds of this confidence interval
    CIresults: results of the confidence interval calculations
    And also all the arguments that may change the result of the estimation.
    """
    popt = fit.popt
    pcov = fit.pcov
    res = fit.res
    infodict = fit.infodict

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

    sorted_yopt, scaled_res, prob, normq = residual_measures(res, yopts)

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
    result["sorted_yopt"] = sorted_yopt
    result["scaled_res"] = scaled_res
    result["normq"] = normq
    result["residuals_evaluation"] = (sorted_yopt, scaled_res, normq)
    result["CI"] = CI
    result["CIs"] = CIs
    #result["CIparams"] = CIparams
    result["CIresults"] = CIresults
    #print "estimate jacobian = %s" % result["extra_output"][-1]["est_jacobian"]
    return ResultStruct(**result)

def residual_measures(res, yest):
    IX = argsort(res)
    scaled_res = res[IX]/std(res)
    sorted_yopt = yest[...,IX]

    prob = (arange(len(scaled_res))+0.5) / len(scaled_res)
    normq = sqrt(2)*erfinv(2*prob-1);

    return sorted_yopt, scaled_res, prob, normq


_restestfields = "res_figure residuals scaled_residuals qqplot dist_residuals"
ResTestResult = namedtuple("ResTestResult", _restestfields)
Plot1dResult = namedtuple("Plot1dResult", "figure estimate data CIs " + _restestfields)

def plot1d(result, loc=0, fig = None, res_fig = None):
    """
    Use matplotlib to display the result of a fit, and return the list of plots used
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
        for p, (low, high) in izip(result.CI,result.CIs[0]):
            l = plot(result.eval_points, low, '--', label='%g%% CI' % (p,))[0]
            h = plot(result.eval_points, high, l.get_color()+'--')[0]
            p_CIs += [l,h]
    if result.param_names:
        param_strs = ", ".join("%s=%g" % (n,v) for n,v in izip(result.param_names, result.popt))
    else:
        param_strs = ", ".join("%g" % v for v in result.popt)
    param_strs = "$%s$" % (param_strs,)

    title("Estimated function %s with params %s" % (result.fct_desc, param_strs))

    xlabel(result.xname)
    ylabel(result.yname)
    legend(loc=loc)

    plots = {"figure": fig, "estimate": p_est, "data": p_data, "CIs": p_CIs}

    plots.update(plot_residual_tests(fct_name="{0} with params {1}".format(result.fct_desc, param_strs), fig = res_fig, **result._asdict())._asdict())

    return Plot1dResult(**plots)

def plot_residual_tests(xdata, yopts, res, fct_name, xname = "X", yname = 'Y', res_name = "residuals",
                        sorted_yopts = None, scaled_res = None, prob = None, normq = None, fig = None,
                        **kwords):
    if fig is None:
        fig = figure()
    else:
        try:
            figure(fig)
        except TypeError:
            figure(fig.number)

    plot1 = subplot(2,2,1)
# First subplot is the residuals
    p_res = plot_residuals(xname, xdata, res_name, res)

    if scaled_res is None or sorted_yopts is None or prob is None or normq is None:
        sorted_yopt, scaled_res, prob, normq = residual_measures(res, yopts)

    plot2 = subplot(2,2,2)
    p_scaled = scaled_location_plot(yname, sorted_yopt, scaled_res)

    subplot(2,2,3)
# Q-Q plot
    qqp = qqplot(scaled_res, normq)

    subplot(2,2,4)
# Distribution of residuals
    drp = plot_dist_residuals(res)

    suptitle("Residual Test for {}".format(fct_name))

    return ResTestResult(fig, p_res, p_scaled, qqp, drp)

def write1d(outfile, result, res_desc, parm_names, CImethod):
    with open(outfile, CSV_WRITE_FLAGS) as f:
        w = csv_writer(f)
        w.writerow(["Function",result.fct.fct.description])
        w.writerow(["Residuals",result.res_name,res_desc])
        w.writerow(["Parameter","Value"])
        for pn, pv in izip(parm_names, result.popt):
            w.writerow([pn, "%.20g" % pv])
        #TODO w.writerow(["Regression Evaluation"])
        w.writerow([])
        w.writerow(["Data"])
        w.writerow([result.xname, result.yname, result.fct_desc, "Residuals: %s" % result.res_name])
        w.writerows(c_[result.xdata, result.ydata, result.yopts, result.res])
        w.writerow([])
        w.writerow(['Model validation'])
        w.writerow([result.yname, 'Normalized residuals', 'Theoretical quantiles'])
        w.writerows(c_[result.sorted_yopt, result.scaled_res, result.normq])
        if result.eval_points is not result.xdata:
            w.writerow([])
            w.writerow(["Interpolated data"])
            w.writerow([result.yname, result.yname])
            w.writerows(c_[result.eval_points, result.interpolation])
        if result.CI:
            w.writerow([])
            w.writerow(["Confidence interval"])
            w.writerow(["Method",CImethod])
            head = ["Parameters"] + list(chain(*[["%g%% - low" % v, "%g%% - high" % v] for v in result.CI]))
            w.writerow(head)
            print result.CIs[1]
            for cis in izip(parm_names, *chain(*result.CIs[1])):
                cistr = [cis[0]] + ["%.20g" % v for v in cis[1:]]
                w.writerow(cistr)
            w.writerow([result.yname])
            head[0] = result.xname
            w.writerow(head)
            w.writerows(c_[tuple(chain([result.eval_points], result.CIs[0][0]))])

def test():
    import residuals
    from numpy.random import rand, randn
    from pylab import plot, savefig, clf, legend, arange, figure, title, show
    from curve_fit import curve_fit

    def test(x,(p0,p1,p2)):
        return p0 + p1*x + p2*x**2

    init = (10,1,1)
    target = (10,4,1.2)
    print "Target parameters: %s" % (target,)
    x = 6*rand(200) - 3
    y = test(x, target)*(1+0.2*randn(x.shape[0]))
    xr = arange(-3, 3, 0.01)
    yr = test(xr,target)

    res = residuals.get('Log residual')

    result = plot_fit(test, x, y, init, eval_points=xr,
                      param_names=("p_0", "p_1", "p_2"), CI=(95,99), fct_desc="$y = p_0 + p_1 x + p_2 x^2$",
                      loc='upper left', fit_kwrds={"residuals":res}, shuffle_args={"add_residual":res.invert},
                      res_desc=res.name)

    result = plot_fit(test, x, y, init, eval_points=xr, shuffle_method=bootstrap.bootstrap_regression,
                      param_names=("p_0", "p_1", "p_2"), CI=(95,99), fct_desc="$y = p_0 + p_1 x + p_2 x^2$",
                      loc='upper left', fit_kwrds={"residuals":res},
                      res_desc=res.name)

    show()
    return locals()

if __name__ == "__main__":
    test()

# /home/barbier/prog/python/curve_fitting/test.csv
