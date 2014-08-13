from __future__ import division, absolute_import, print_function
import numpy as np


def local_linear_1d(bw, xdata, ydata, points, kernel, out):
    r'''
    We are trying to find the fitting for points :math:`x` given a gaussian kernel
    Given the following definitions:

    .. math::

        x_0 &=& x-x_i

        \begin{array}{rlc|rlc}
        w_i &=& \mathcal{K}\left(\frac{x_0}{h}\right) & W &=& \sum_i w_i \\
        X &=& \sum_i w_i x_0 & X_2 &=& w_i x_0^2 \\
        Y &=& \sum_i w_i y_i & Y_2 &=& \sum_i w_i y_i x_0
        \end{array}



    The fitted value is given by:

    .. math::

        f(x) = \frac{X_2 T - X Y_2}{W X_2 - X^2}

    '''
    x0 = points - xdata[:, np.newaxis]
    x02 = x0 * x0
    # wi = kernel(x0 / bw)
    wi = np.exp(-x02 / (2.0 * bw * bw))
    X = np.sum(wi * x0, axis=0)
    X2 = np.sum(wi * x02, axis=0)
    wy = wi * ydata[:, np.newaxis]
    Y = np.sum(wy, axis=0)
    Y2 = np.sum(wy * x0, axis=0)
    W = np.sum(wi, axis=0)
    return None, np.divide(X2 * Y - Y2 * X, W * X2 - X * X, out)
