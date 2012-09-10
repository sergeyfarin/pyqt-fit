__author__ = "Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>"

import sys
from ..path import path

sys_modules = "*.so"
if sys.platform == 'win32' or sys.platform == 'cygwin':
    sys_modules = "*.dll"
elif sys.platform == 'darwin':
    sys_modules = "*.dylib"

class Function(object):
    def __init__(self, fct, name, desc, args, init_args):
        self.fct = fct
        self.name = name
        self.description = desc
        self.args = args
        self.init_args = init_args

    def __call__(self, *args, **kwords):
        return self.fct(*args, **kwords)

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
            if name and desc and args and init_args:
                result[name] = Function(attr, name, desc, args, init_args)
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
    fct: callable object
        Function of interest. The callable also has the following properties:
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
    """
    return functions.get(name, None)

def names():
    """
    Return the list of names for the functions
    """
    return functions.keys()

