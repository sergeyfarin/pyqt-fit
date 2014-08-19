try:
    from .cy_binning import fast_linbin as fast_bin
except ImportError:
    from .py_binning import fast_bin

def usePython():
    global fast_bin
    from .py_linbin import fast_bin

def useCython():
    global fast_bin
    from .cy_linbin import fast_linbin as fast_bin

