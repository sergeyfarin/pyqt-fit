import sharedmem

def initialize_shared(nx, ny, result_array, shuffled_x, shuffled_y, eval_points, fit, fit_args, fit_kwrds):
    initialize(nx, ny, result_array.np, shuffled_x.np, shuffled_y.np, eval_points.np, fit, fit_args, fit_kwrds)

def initialize(nx, ny, result_array, shuffled_x, shuffled_y, eval_points, fit, fit_args, fit_kwrds):
    globals().update(locals())

def bootstrap_result(worker, start_repeats, end_repeats):
    #print "Starting worker {} from {} to {}".format(worker, start_repeats, end_repeats)
    try:
        for i in xrange(start_repeats, end_repeats):
            #print "Worker {} runs iteration {} with fit: {}".format(worker, i, fit)
            new_fit = fit(shuffled_x[...,i%nx,:], shuffled_y[i%ny,:], *fit_args, **fit_kwrds)
            #print "new_fit = {}".format(new_fit)
            result_array[i+1] = new_fit(eval_points)
    except Exception, ex:
        print "Error, exception caught: {}".format(ex)
    #print "Worker {} finished".format(worker)

