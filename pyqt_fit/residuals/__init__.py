from __future__ import print_function

__author__ = "Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>"

from ..utils import namedtuple
from .. import loader
from path import path
import os

_fields = ['name', 'description', 'invert', 'Dfun', '__call__']

residuals = None

Residual = namedtuple('Residual', ['fct', 'name', 'description', 'invert', 'Dfun', '__call__'])


def find_functions(module):
    """
    Find and all the residual functions defined in the given module
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
    Load and register all the residual functions available.

    It will be looking in the current folder, but also in the "residuals"
    subfolders of the paths defined in the PYQTFIT_PATH environment
    variable.
    """
    global residuals
    residuals = loader.load(find_functions)
    extra_path = os.environ.get("PYQTFIT_PATH", "").split(":")
    for ep in extra_path:
        ep = path(ep)
        if ep and (ep / "residuals").exists():
            residuals.update(loader.load(find_functions, ep / "residuals"))
    return residuals

load()


def get(name):
    """
    Return the description of the residuals names 'name'

    Parameters
    ----------
    name: str
        Name of the residuals

    Returns
    -------
    fct: callable object
        Callable object that will compute the residuals. The callable has the following attributes:
        fct: callable
            Function computing the residuals
        name: str
            Name of the residuals type
        description: str
            Formula of the residual
        invert: callable
            Function to add residuals to a data set
    """
    return residuals.get(name, None)


def names():
    """
    List the names of available residuals
    """
    return residuals.keys()
