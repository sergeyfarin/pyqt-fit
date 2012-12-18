__author__ = "Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>"

from ..utils import namedtuple
from .. import loader
from path import path
import os

Residual = namedtuple('Residual', ['fct', 'name', 'description', 'invert', 'Dfun', '__call__'])

def find_functions(module):
    content = dir(module)
    result = {}
    for c in content:
        attr = getattr(module, c)
        if hasattr(attr, '__doc__'):
            doc = getattr(attr, '__doc__')
            if doc is None:
                continue
            doc = doc.split('\n')
            name = None
            desc = None
            invert = None
            dfun = None
            for l in doc:
                l = l.strip()
                fields = l.split(':', 1)
                if len(fields) == 2:
                    if fields[0] == 'Name':
                        name = fields[1].strip()
                    elif fields[0] == 'Invert':
                        inv_name = fields[1].strip()
                        if hasattr(module, inv_name):
                            invert = getattr(module, inv_name)
                            if not callable(invert):
                                invert = None
                    elif fields[0] == 'Formula':
                        desc = fields[1].strip()
                    elif fields[0] == 'Dfun':
                        dfun_name = fields[1].strip()
                        if hasattr(module, dfun_name):
                            dfun = getattr(module, dfun_name)
                            if not callable(dfun):
                                dfun = None
            if name and desc and invert: # dfun is optional
                result[name] = Residual(attr, name, desc, invert, dfun, attr)
    return result

def load():
    global residuals
    residuals = loader.load(find_functions)
    extra_path = os.environ.get("PYQTFIT_PATH", "").split(":")
    for ep in extra_path:
        ep = path(ep)
        if ep and (ep/"residuals").exists():
            residuals.update(loader.load(find_functions, ep/"residuals"))
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
    return residuals.get(name,None)

def names():
    """
    List the names of available residuals
    """
    return residuals.keys()

