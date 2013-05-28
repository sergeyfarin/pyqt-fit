from __future__ import print_function

__author__ = "Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>"

from ..utils import namedtuple
from .. import loader
import os
from path import path

_fields = ['name', 'description', 'args', 'init_args', 'Dfun', '__call__']

functions = None

Function = namedtuple('Function', _fields)


def find_functions(module):
    """
    Find and register the functions defined in the given module
    """
    content = dir(module)
    result = {}
    for c in content:
        obj = getattr(module, c)
        try:
            if isinstance(obj, type):
                obj = obj()
            for attr in _fields:
                if not hasattr(obj, attr):
                    break
            else:
                result[obj.name] = obj
        except Exception as ex:  # Silently ignore any exception
            print("Error: '{}'".format(ex))
            pass
    return result


def load():
    """
    Find and register the all the functions available.

    It will be looking in the current folder, but also in the "functions"
    subfolders of the paths specified in the PYQTFIT_PATH environment variable.
    """
    global functions
    functions = loader.load(find_functions)
    extra_path = os.environ.get("PYQTFIT_PATH", "").split(":")
    for ep in extra_path:
        ep = path(ep)
        if ep and (ep / "functions").exists():
            functions.update(loader.load(find_functions, ep / "functions"))
    return functions

load()


def get(name):
    """
    Get a tuple representing the function or None if not found

    Parameters
    ----------
    name: str
        Name of the function to find

    Returns
    -------
    fct: namedtuple
        Function of interest. The named tuple also has the following fields:
        fct: callable
            The function itself
        name: string
            Name of the function
        description: string
            Description (i.e. formula) of the function
        args: tuple of string
            List of argument names
        init_args: callable
            Function used to estimate initial parameters from dataset
        Dfun: callable
            Jacobian of the function (in column, so call leastsq with col_deriv=1)
    """
    f = functions.get(name, None)
    #print("Getting function '{}'".format(name))
    #if f is not None:
        #print("  Dfun = {}".format(f.Dfun))
    return f


def names():
    """
    Return the list of names for the functions
    """
    return functions.keys()
