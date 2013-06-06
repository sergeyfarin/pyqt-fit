"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module specifically implement the curve fitting, wrapping the default
scipy.optimize.leastsq function. It allows for parameter value fixing,
different kind of residual and added constraints function.
"""

from __future__ import division, print_function, absolute_import
from scipy import optimize
from numpy import array, inf
from .compat import lrange


class CurveFitting(object):
    r"""
    Fit a curve using the :py:func:`scipy.optimize.leastsq` function

    :type  xdata: ndarray
    :param xdata: Explaining values

    :type  ydata: ndarray
    :param ydata: Target values

    :type  p0: tuple
    :param p0: Initial estimates for the parameters of fct

    :type  fct: callable
    :param fct: Function to optimize. The call will be equivalent
        to ``fct(p0, xdata, *args)``

    :type  args: tuple
    :param args: Additional arguments for the function

    :type  residuals: callable or None
    :param residuals: Function computing the residuals. The call is equivalent
        to ``residuals(y, fct(x))`` and it should return a ndarray. If None,
        residuals are simply the difference between the computed and expected
        values.

    :type  fix_params: tuple of int
    :param fix_params: List of indices for the parameters in p0 that shouldn't
        change

    :type  Dfun: callable
    :param Dfun: Function computing the jacobian of fct w.r.t. the parameters.
        The call will be equivalent to ``Dfun(p0, xdata, *args)``

    :type  Dres: callable
    :param Dres: Function computing the jacobian of the residuals w.r.t. the
        parameters. The call will be equivalent to
        ``Dres(y, fct(x), DFun(x))``. If None, residuals must also be None.

    :type  col_deriv: int
    :param col_deriv: Define if Dfun returns the derivatives by row or column.
        With n = len(xdata) and m = len(p0), the shape of output of Dfun must
        be (n,m) if 0, and (m,n) if non-0.

    :type  constraints: callable
    :param constraints: If not None, this is a function that should always
        return a list ofvalues (the same), to add penalties for bad parameters.
        The function call is equivalent to: ``constraints(p0)``

    :type  lsq_args: tuple
    :param lsq_args: List of unnamed arguments passed to ``optimize.leastsq``,
        starting with ``ftol``

    :type  lsq_kword: dict
    :param lsq_kword: Dictionnary of named arguments passed to
        py:func:`scipy.optimize.leastsq`, starting with ``ftol``

    Once constructed, the following variables contain the result of
    the fitting:

    :ivar ndarray popt: The solution (or the result of the last iteration for
        an unsuccessful call)
    :ivar ndarray pcov: The estimated covariance of popt.  The diagonals
        provide the variance of the parameter estimate.
    :ivar ndarray res: Final residuals

    :ivar dict infodict: a dictionary of outputs with the keys:

            ``nfev``
                the number of function calls
            ``fvec``
                the function evaluated at the output
            ``fjac``
                A permutation of the R matrix of a QR factorization of
                the final approximate Jacobian matrix, stored column wise.
                Together with ipvt, the covariance of the estimate can be
                approximated.
            ``ipvt``
                an integer array of length N which defines a permutation
                matrix, ``p``, such that ``fjac*p = q*r``, where ``r`` is upper
                triangular with diagonal elements of nonincreasing
                magnitude. Column ``j`` of ``p`` is column ``ipvt(j)`` of the
                identity matrix.
            ``qtf``
                the vector ``(transpose(q) * fvec)``
            ``CI``
                list of tuple of parameters, each being the lower and
                upper bounds for the confidence interval in the CI
                argument at the same position.
            ``est_jacobian``
                True if the jacobian is estimated, false if the
                user-provided functions have been used

    .. note::

        In this implementation, residuals are supposed to be a generalisation
        of the notion of difference. In the end, the mathematical expression
        of this minimisation is:

        .. math::

            \hat{\theta} = \argmin_{\theta\in \mathbb{R}^p}
                \sum_i r(y_i, f(\theta, x_i))^2

        Where :math:`\theta` is the vector of :math:`p` parameters to optimise,
        :math:`r` is the residual function and :math:`f` is the function being
        fitted.
    """

    def __init__(self, xdata, ydata, p0, fct, args=(), residuals=None,
                 fix_params=(), Dfun=None, Dres = None, col_deriv=1,
                 constraints = None, *lsq_args, **lsq_kword):
        self.fct = fct
        if residuals is None:
            residuals = lambda x, y: (x - y)
            Dres = lambda y1, y0, dy: -dy

        use_derivs = (Dres is not None) and (Dfun is not None)
        df = None

        if fix_params:
            fix_params = tuple(fix_params)
            p_save = array(p0, dtype=float)
            change_params = lrange(len(p0))
            try:
                for i in fix_params:
                    change_params.remove(i)
            except ValueError:
                raise ValueError("List of parameters to fix is incorrect: "
                                 "contains either duplicates or values "
                                 "out of range.")
            p0 = p_save[change_params]

            def f(p, *args):
                p1 = array(p_save)
                p1[change_params] = p
                y0 = fct(p1, xdata, *args)
                return residuals(ydata, y0)
            if use_derivs:
                def df(p, *args):
                    p1 = array(p_save)
                    p1[change_params] = p
                    y0 = fct(p1, xdata, *args)
                    dfct = Dfun(p1, xdata, *args)
                    result = Dres(ydata, y0, dfct)
                    if col_deriv != 0:
                        return result[change_params]
                    else:
                        return result[:, change_params]
                    return result
        else:

            def f(p, *args):  # noqa
                y0 = fct(p, xdata, *args)
                return residuals(ydata, y0)
            if use_derivs:

                def df(p, *args):  # noqa
                    dfct = Dfun(p, xdata, *args)
                    y0 = fct(p, xdata, *args)
                    return Dres(ydata, y0, dfct)

        optim = optimize.leastsq(f, p0, args, full_output=1, Dfun=df,
                                 col_deriv=col_deriv, *lsq_args, **lsq_kword)
        popt, pcov, infodict, mesg, ier = optim
        #infodict['est_jacobian'] = not use_derivs

        if fix_params:
            p_save[change_params] = popt
            popt = p_save

        if not ier in [1, 2, 3, 4]:
            raise RuntimeError("Unable to determine number of fit parameters. "
                               "Error returned by scipy.optimize.leastsq:\n%s"
                               % (mesg,))

        res = residuals(ydata, fct(popt, xdata, *args))
        if (len(res) > len(p0)) and pcov is not None:
            s_sq = (res ** 2).sum() / (len(ydata) - len(p0))
            pcov = pcov * s_sq
        else:
            pcov = inf

        self.popt = popt
        self.pcov = pcov
        self.res = res
        self.infodict = infodict

    def __call__(self, xdata):
        """
        Return the value of the fitted function for each of the points in
        ``xdata``
        """
        return self.fct(self.popt, xdata)
