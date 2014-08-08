from __future__ import division, print_function, absolute_import
import sys
import traceback
from .compat import irange, izip

nx = ny = 1
shuffled_x = shuffled_y = None
fit_args = ()
fit_kwrds = {}
fit = None
result_array = None
eval_points = None
extra_arrays = None
extra_attrs = None


def initialize_shared(nx, ny, result_array, extra_arrays, shuffled_x,
                      shuffled_y, eval_points, extra_attrs, fit,
                      fit_args, fit_kwrds):
    initialize(nx, ny, result_array.np, [ea.np for ea in extra_arrays],
               shuffled_x.np, shuffled_y.np, eval_points.np, extra_attrs,
               fit, fit_args, fit_kwrds)


def initialize(nx, ny, result_array, extra_arrays, shuffled_x, shuffled_y,
               eval_points, extra_attrs, fit, fit_args, fit_kwrds):
    globals().update(locals())


def bootstrap_result(worker, start_repeats, end_repeats):
    #print("Starting worker {} from {} to {}".format(worker, start_repeats, end_repeats))
    try:
        for i in irange(start_repeats, end_repeats):
            #print("Worker {} runs iteration {} with fit: {}".format(worker, i, fit))
            new_fit = fit(shuffled_x[..., i % nx, :], shuffled_y[i % ny, :],
                          *fit_args, **fit_kwrds)
            new_fit.fit()
            #print("new_fit = {}".format(new_fit))
            result_array[i + 1] = new_fit(eval_points)
            for ea, attr in izip(extra_arrays, extra_attrs):
                ea[i + 1] = getattr(new_fit, attr)
    except Exception:
        traceback.print_exc(None, sys.stderr)
        raise
    #print "Worker {} finished".format(worker)
