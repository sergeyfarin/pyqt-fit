__author__ = "Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>"

import sys
from ..path import path
import traceback
from ..utils import namedtuple

sys_modules = "*.so"
if sys.platform == 'win32' or sys.platform == 'cygwin':
    sys_modules = "*.dll"
elif sys.platform == 'darwin':
    sys_modules = "*.dylib"

Function = namedtuple('Function', ['fct', 'name', 'description', 'args', 'init_args', 'Dfun', '__call__'])

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
            args = None
            dfun = None
            init_args = None
            for l in doc:
                l = l.strip()
                fields = l.split(':', 1)
                if len(fields) == 2:
                    if fields[0] == 'Name':
                        name = fields[1].strip()
                    elif fields[0] == 'Parameters':
                        args = fields[1].strip().split()
                    elif fields[0] == 'Function':
                        desc = fields[1].strip()
                    elif fields[0] == 'ParametersEstimate':
                        params_name = fields[1].strip()
                        if hasattr(module, params_name):
                            init_args = getattr(module, params_name)
                            if not callable(init_args):
                                init_args = None
                    elif fields[0] == 'Dfun':
                        dfun_name = fields[1].strip()
                        if hasattr(module, dfun_name):
                            dfun = getattr(module, dfun_name)
                            if not callable(dfun):
                                dfun = None
            if name and desc and args and init_args: # dfun is optional
                result[name] = Function(attr, name, desc, args, init_args, dfun, attr)
    return result

def load():
    system_files = [ __file__ ]
    sys_files = set()
    for f in system_files:
        if f.endswith(".pyo") or f.endswith(".pyc"):
            f = f[:-3]+"py"
        sys_files.add(path(f).abspath())
    search_path = path(__file__).abspath().dirname()
    fcts = {}
# Search for python, cython and modules
    for f in (search_path.files("*.py") + search_path.files("*.pyx") + search_path.files(sys_modules)):
        if f not in sys_files:
            module_name = f.namebase
            pack_name = 'functions.%s' % module_name
            try:
                mod = sys.modules.get(pack_name)
                if mod:
                    reload(mod)
                else:
                    exec "import %s" % module_name in globals()
                    mod = sys.modules.get(pack_name)
                mod = eval(module_name)
                fcts.update(find_functions(mod))
            except ImportError:
                print "Warning, cannot import module '%s'" % (module_name,)
    global functions
    functions = fcts
    return fcts

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
    print "Getting function '%s'" % name
    if f is not None:
        print "  Dfun = %s" % f.Dfun
    return f

def names():
    """
    Return the list of names for the functions
    """
    return functions.keys()

