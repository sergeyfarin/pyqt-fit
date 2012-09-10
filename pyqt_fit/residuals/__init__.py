__author__ = "Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>"

import sys
from ..path import path

sys_modules = "*.so"
if sys.platform == 'win32' or sys.platform == 'cygwin':
    sys_modules = "*.dll"
elif sys.platform == 'darwin':
    sys_modules = "*.dylib"

class Residual(object):
    def __init__(self, fct, name, desc, invert):
        self.fct = fct
        self.name = name
        self.description = desc
        self.invert = invert

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
            invert = None
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
            if name and desc and invert:
                result[name] = Residual(attr, name, desc, invert)
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
    for f in (search_path.files("*.py") + search_path.files("*.pyx") + search_path.files(sys_modules)):
        if f not in sys_files:
            module_name = f.namebase
            pack_name = 'residuals.%s' % module_name
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
    global residuals
    residuals = fcts
    return fcts

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

