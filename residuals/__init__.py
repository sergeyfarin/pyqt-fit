__author__ = "Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>"

import sys
from path import path
import traceback
import imp

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
                attr.name = name
                attr.invert = invert
                attr.description = desc
                result[name] = attr
    return result

def load():
    system_files = [ __file__ ]
    sys_files = set()
    for f in system_files:
        if f.endswith(".pyo") or f.endswith(".pyc"):
            f = f[:-3]+"py"
        sys_files.add(path(f).abspath())
    search_path = path(__file__).abspath().dirname()
    errors = []
    fcts = {}
    for f in search_path.files("*.py"):
        if f not in sys_files:
            module_name = f.basename()[:-3]
            full_name = "residuals."+module_name
            fd, pathname, desc = imp.find_module(module_name, [f.dirname()])
            try:
                module = imp.load_module(full_name, fd, pathname, desc)
                fcts.update(find_functions(module))
            except Exception, ex:
                tb = sys.exc_info()[2]
                error_loc = "\n".join("In file %s, line %d\n\tIn '%s': %s" % e for e in traceback.extract_tb(tb))
                errors.append((f,"Exception %s:\n%s\n%s" % (type(ex).__name__, str(ex), error_loc)))
            finally:
                if fd:
                    fd.close()
    if errors:
        errs =  "\n\n".join("In file %s:\n%s" % (f,e) for f,e in errors)
        raise RuntimeError("Errors while loading modules:\n'%s'" % errs)
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
    fct: callable
        Callable object that will compute the residuals. The callable has the following attributes:
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

