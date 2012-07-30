from __future__ import division
from curve_fit import curve_fit
from numpy import sort, iterable, argsort, std, abs, sqrt, arange, pi
from pylab import figure, title, legend, plot, xlabel, ylabel, subplot, clf, ylim, hist, suptitle, gca
import bootstrap
from itertools import izip
from scipy.special import erfinv, gamma
from scipy import stats
from average_data import SpatialAverage

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
    av = SpatialAverage(xdata, res)
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

def scaled_location_plot(xname, xdata, scaled_res):
    """
    Plot the scaled location, given the X and scaled residuals
    """
    ydata = sqrt(abs(scaled_res))
    p_scaled = plot(xdata, ydata, '+')[0]
    av = SpatialAverage(xdata, ydata)
    xr = arange(xdata.min(), xdata.max(), (xdata.max() - xdata.min())/1024)
    rr = av(xr)
    p_smooth = plot(xr, rr, 'g')[0]
    expected_mean = 2**(1/4)*gamma(3/4)/sqrt(pi)
    plot([xdata.min(), xdata.max()], [expected_mean, expected_mean], 'r--')
    title('Scale-location')
    xlabel(xname)
    ylabel('$|$Normalized residuals$|^{1/2}$')
    gca().set_yticks([0,1,2])
    return [p_scaled, p_smooth]

def qqplot(scaled_res):
    """
    Draw a Q-Q Plot from the sorted, scaled residuals (i.e. residuals sorted
    and normalized by their standard deviation)
    """
    prob = (arange(len(scaled_res))+0.5) / len(scaled_res)
    normq = sqrt(2)*erfinv(2*prob-1);
    qqp = []
    qqp += plot(normq, scaled_res, '+');
    qqp += plot(normq, normq, 'r--');
    xlabel('Theoretical quantiles');
    ylabel('Normalized residuals');
    title('Normal Q-Q plot');
    return normq, qqp

class ResultStruct(object):
    pass

def plot_fit(fct, xdata, ydata, p0, fit = curve_fit, eval_points=None,
             CI=(), xname="X", yname="Y", fct_desc = None, param_names = (), res_desc = 'Standard',
             args=(), loc=None, **kwrds):
    """
    Fit the function ``fct(xdata, p0, *args)`` using the ``fit`` function

    Parameters
    ----------
    fct: callable
        Function to fit the call must be ``fct(xdata, p0, *args)``
    xdata: ndarray of shape (N,) or (k,N) for function with k prefictors
        The independent variable where the data is measured
    ydata: ndarray
        The dependant data
    p0: ndarray
        Initial estimate of the parameters
    fit: callable
        Function to use for the estimation. The call is ``fit(fct, xdata,
        ydata, p0, args=args, **kwrds)``. The three first returned values
        must be: the best parameters found, the covariance of the parameters,
        and the residuals with these parameters
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
    res_desc: string
        String describing the residuals
    args: tuple
        Extra arguments for fct
    loc: string or int
        Location of the legend in the graphs
    kwrds: dict
        Extra named arguments are forwarded to the bootstrap or fit function,
        depending on which is called

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
    CIparams: list of pair of ndarray
        For each CI value, a pair of ndarray is provided for the lower and
        upper bound of the parameters
    plots: dict
        Dictionnary to access the various plot curves
    extra_output: extra output provided by the fit or bootstrap function
    """
    CIs = []
    CIparams = []
    if CI:
        if not iterable(CI):
            CI = (CI,)
        result = bootstrap.bootstrap(fct, xdata, ydata, p0, CI, args=args, eval_points=eval_points, fit=fit, **kwrds)
        popt, pcov, res, CIs, CIparams = result[:5]
        extra_output = result[5:]
    else:
        result = fit(fct, xdata, ydata, p0, args=args, **kwrds)
        popt, pcov, res = result[:3]
        extra_output = result[3:]

    figure()
    clf()
    yopts = fct(xdata, popt, *args)
    yvals = fct(eval_points, popt, *args)
    p_est = plot(eval_points, yvals, label='estimated')[0]
    p_data = plot(xdata, ydata, '+', label='data')[0]
    p_CIs = []
    if CI:
        for p, (low, high) in izip(CI,CIs):
            l = plot(eval_points, low, '--', label='%g%% CI' % (p,))[0]
            h = plot(eval_points, high, l.get_color()+'--')[0]
            p_CIs += [l,h]
    if param_names:
        param_strs = ", ".join("%s=%g" % (n,v) for n,v in izip(param_names, popt))
    else:
        param_strs = ", ".join("%g" % v for v in popt)
    param_strs = "$%s$" % (param_strs,)
    title("Estimated function %s with params %s" % (fct_desc, param_strs))
    xlabel(xname)
    ylabel(yname)
    legend(loc=loc)

    figure()
    plot1 = subplot(2,2,1)
# First subplot is the residuals
    p_res = plot_residuals(xname, xdata, res_desc, res)

    plot2 = subplot(2,2,2)
# Scaled location
    IX = argsort(res)
    scaled_res = res[IX]/std(res)
    sorted_x = xdata[...,IX]
    p_scaled = scaled_location_plot(xname, sorted_x, scaled_res)

    subplot(2,2,3)
# Q-Q plot
    normq, qqp = qqplot(scaled_res)

    subplot(2,2,4)
# Distribution of residuals
    plot_dist_residuals(res)

    plots = {"estimate": p_est, "data": p_data, "CIs": p_CIs, "residuals": p_res, "scaled residuals": p_scaled, "qqplot": qqp}

    suptitle("Checks of correctness for function %s with params %s" % (fct_desc, param_strs))

    result = ResultStruct()
    result.popt = popt
    result.res = res
    result.yopts = yopts
    result.interpolation = (eval_points, yvals)
    result.residuals_evaluation = (sorted_x, scaled_res, normq)
    result.CIs = CIs
    result.CIparams = CIparams
    result.plots = plots
    result.extra_output = extra_output
    return result

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
                      loc='upper left', fit_args={"residuals":res}, shuffle_args={"add_residual":res.invert},
                      res_desc=res.name)

    result = plot_fit(test, x, y, init, eval_points=xr, shuffle_method=bootstrap.bootstrap_regression,
                      param_names=("p_0", "p_1", "p_2"), CI=(95,99), fct_desc="$y = p_0 + p_1 x + p_2 x^2$",
                      loc='upper left', fit_args={"residuals":res},
                      res_desc=res.name)

    show()
    return locals()

if __name__ == "__main__":
    test()

