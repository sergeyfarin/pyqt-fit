Module ``pyqt_fit.bootstrap``
=============================

.. automodule:: pyqt_fit.bootstrap

.. currentmodule:: pyqt_fit.bootstrap

Bootstrap Shuffling Methods
---------------------------

.. autofunction:: bootstrap_residuals

.. autofunction:: bootstrap_regression

Main Boostrap Functions
-----------------------

.. autofunction:: bootstrap

.. class:: BootstrapResult(y_fit, y_est, y_eval, CIs, shuffled_xs, shuffled_ys, full_results)

  .. note::

    This is a class created with :py:func:`pyqt_fit.utils.namedtuple`.

  .. py:attribute:: y_fit

            Y estimated on xdata

  .. py:attribute:: y_est: ndarray

            Y estimated on eval_points

  .. py:attribute:: CIs

            List of confidence intervals. The first element is for the estimated values
            on ``eval_points``. The others are for the extra attributes specified in
            ``extra_attrs``. Each array is a 3-dimensional array (Q,2,N), where
            Q is the number of confidence interval and N is the number of data
            points. Values (x,0,y) give the lower bounds and (x,1,y) the upper
            bounds of the confidence intervals.

  .. py:attribute:: shuffled_xs

            if full_results is True, the shuffled x's used for the bootstrapping

  .. py:attribute:: shuffled_ys

            if full_results is True, the shuffled y's used for the bootstrapping

  .. py:attribute:: full_results

            if full_results is True, the estimated y's for each shuffled_ys


