"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module specifically implement the curve fitting, wrapping the default
scipy.optimize.leastsq function. It allows for parameter value fixing,
different kind of residual and added constraints function.
"""

from __future__ import division, print_function, absolute_import
from scipy import optimize
from .compat import lrange
import numpy as np


class CurveFitting(object):
    r"""
    Fit a curve using the :py:func:`scipy.optimize.leastsq` function

    :type  xdata: ndarray
    :param xdata: Explaining values

    :type  ydata: ndarray
    :param ydata: Target values

    Once fitted, the following variables contain the result of
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

    def __init__(self, xdata, ydata, **kwords):
        self._fct = None
        self._Dfun = None
        self._residuals = None
        self._Dres = None
        self._col_deriv = True
        self._constraints = None
        self._lsq_args = ()
        self._lsq_kwords = {}
        self._xdata = None
        self._ydata = None
        self._p0 = None
        self._fix_params = None

        self.xdata = xdata
        self.ydata = ydata

        self._fitted = False

        for n in kwords:
            setattr(self, n, kwords[n])

        if self._residuals is None:
            self._residuals = lambda x, y: (x - y)
            self._Dres = lambda y1, y0: -1

    def need_fit(self):
        """
        Function to be called if the object need to be fitted again
        """
        self._fitted = False

    @property
    def fitted(self):
        """
        Check if the object has been fitted or not
        """
        return self._fitted

    @property
    def function(self):
        """
        Function to be fitted. The call of the function will be::

            function(params, xs)
        """
        return self._fct

    @function.setter
    def function(self, f):
        self.need_fit()
        self._fct = f

    @property
    def Dfun(self):
        """
        Jacobian of the function with respect to its parameters.

        :Note: col_deriv defines if the derivative with respect to a given parameter is in column or row

        If not provided, a numerical approximation will be used instead.
        """
        return self._Dfun

    @Dfun.setter
    def Dfun(self, df):
        self.need_fit()
        self._Dfun = df

    @Dfun.deleter
    def Dfun(self):
        self.need_fit()
        self._Dfun = None

    @property
    def col_deriv(self):
        """
        Define if Dfun returns the derivatives by row or column.

        If ``col_deriv`` is ``True``, each line correspond to a parameter and each column to a point.
        """
        return self._col_deriv

    @col_deriv.setter
    def col_deriv(self, value):
        self._col_deriv = bool(value)
        self.need_fit()

    @property
    def residuals(self):
        """
        Residual function to use. The call will be::

            residuals(y_measured, y_est)

        where ``y_measured`` are the estimated values and ``y_est`` the measured ones.

        :Default: the defauls is ``y_measured - y_est``
        """
        return self._residuals

    @residuals.setter
    def residuals(self, f):
        self.need_fit()
        self._residuals = f

    @property
    def Dres(self):
        """
        Derivative of the residual function with respec to the estimated values. The call will be:

            Dres(y_measured, y_est)

        :Default: as the default residual is ``y_measured - y_est``, the default derivative is ``-1``
        """
        return self._Dres

    @Dres.setter
    def Dres(self, df):
        self.need_fit()
        self._Dres = df

    @Dres.deleter
    def Dres(self):
        self.need_fit()
        self._Dres = None

    @property
    def lsq_args(self):
        """
        Extra arguments to give to the least-square algorithm.

        See :py:func:`scipy.optimize.leastsq` for details
        """
        return self._lsq_args

    @lsq_args.setter
    def lsq_args(self, val):
        self.need_fit()
        self._lsq_args = tuple(val)

    @lsq_args.deleter
    def lsq_args(self):
        self._lsq_args = ()

    @property
    def lsq_kwords(self):
        """
        Extra named arguments to give to the least-square algorithm.

        See :py:func:`scipy.optimize.leastsq` for details
        """
        return self._lsq_kwords

    @lsq_kwords.setter
    def lsq_kwords(self, val):
        self.need_fit()
        self._lsq_kwords = dict(val)

    @lsq_kwords.deleter
    def lsq_kwords(self):
        self._lsq_kwords = {}

    @property
    def xdata(self):
        """
        Explaining values.
        """
        return self._xdata

    @xdata.setter
    def xdata(self, value):
        value = np.atleast_1d(value).squeeze()
        assert len(value.shape) < 3, "Error, xdata must be at most a 2D array"
        self._xdata = value
        self.need_fit()

    @property
    def ydata(self):
        """
        Target values.
        """
        return self._ydata

    @ydata.setter
    def ydata(self, value):
        value = np.atleast_1d(value).squeeze()
        assert len(value.shape) == 1, "Error, ydata must be at most a 1D array"
        self._ydata = value
        self.need_fit()

    @property
    def p0(self):
        """
        Initial fitting parameters
        """
        return self._p0

    @p0.setter
    def p0(self, value):
        value = np.atleast_1d(value)
        assert len(value.shape) == 1, "Error, p0 must be at most a 1D array"
        self._p0 = value

    @property
    def constraints(self):
        """
        Function returning additional constraints to the problem
        """
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        assert callable(value), "Error, constraints must be a callable returning a 1d array"
        self._constraints = value

    @constraints.deleter
    def constraints(self):
        self._constraints = None

    @property
    def fix_params(self):
        """
        Index of parameters that shouldn't be touched by the algorithm
        """
        return self._fix_params

    @fix_params.setter
    def fix_params(self, value):
        self._fix_params = tuple(value)

    @fix_params.deleter
    def fix_params(self):
        self._fix_params = None

    def fit(self):
        """
        Fit the curve
        """
        Dres = self.Dres
        Dfun = self.Dfun
        fct = self.function
        residuals = self.residuals
        col_deriv = self.col_deriv
        p0 = self.p0
        xdata = self.xdata
        ydata = self.ydata
        fix_params = self.fix_params

        use_derivs = (Dres is not None) and (Dfun is not None)
        df = None
        f = None

        if fix_params:
            p_save = np.array(p0, dtype=float)
            change_params = lrange(len(p0))
            try:
                for i in fix_params:
                    change_params.remove(i)
            except ValueError:
                raise ValueError("List of parameters to fix is incorrect: "
                                 "contains either duplicates or values "
                                 "out of range.")
            p0 = p_save[change_params]

            def f_fixed(p):
                p1 = np.array(p_save)
                p1[change_params] = p
                y0 = fct(p1, xdata)
                return residuals(ydata, y0)
            f = f_fixed
            if use_derivs:
                def df_fixed(p):
                    p1 = np.array(p_save)
                    p1[change_params] = p
                    y0 = fct(p1, xdata)
                    dfct = Dfun(p1, xdata)
                    dr = Dres(ydata, y0)
                    if col_deriv:
                        return dfct[change_params]*dr
                    return dfct[:,change_params]*dr[:, np.newaxis]
                df = df_fixed
        else:
            def f_free(p):
                y0 = fct(p, xdata)
                return residuals(ydata, y0)
            f = f_free
            if use_derivs:
                def df_free(p):
                    dfct = Dfun(p, xdata)
                    y0 = fct(p, xdata)
                    dr = np.atleast_1d(Dres(ydata, y0))
                    if col_deriv:
                        return dfct*dr
                    return dfct*dr[:, np.newaxis]
                df = df_free

        if use_derivs:
            self.df = df

        cd = 1 if col_deriv else 0
        optim = optimize.leastsq(f, p0, full_output=1, Dfun=df,
                                 col_deriv=cd, *self.lsq_args, **self.lsq_kwords)
        popt, pcov, infodict, mesg, ier = optim
        #infodict['est_jacobian'] = not use_derivs

        if fix_params:
            p_save[change_params] = popt
            popt = p_save

        if not ier in [1, 2, 3, 4]:
            raise RuntimeError("Unable to determine number of fit parameters. "
                               "Error returned by scipy.optimize.leastsq:\n%s"
                               % (mesg,))

        res = residuals(ydata, fct(popt, xdata))
        if (len(res) > len(p0)) and pcov is not None:
            s_sq = (res ** 2).sum() / (len(ydata) - len(p0))
            pcov = pcov * s_sq
        else:
            pcov = np.inf

        self.popt = popt
        self.pcov = pcov
        self.res = res
        self.infodict = infodict
        self._fitted = True

    def __call__(self, xdata):
        """
        Return the value of the fitted function for each of the points in
        ``xdata``
        """
        if not self.fitted:
            self.fit()
        return self.function(self.popt, xdata)

