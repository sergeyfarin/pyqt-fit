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
            args = None
            parms = None
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
                            parms = getattr(module, params_name)
                            if not callable(parms):
                                parms = None
            if name and desc and args and parms:
                attr.name = name
                attr.description = desc
                attr.args = args
                attr.parms = parms
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
            full_name = "functions."+module_name
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
    fct: callable
        Function of interest. The callable also has the following properties:
        name: string
            Name of the function
        description: string
            Description (i.e. formula) of the function
        args: tuple of string
            List of argument names
        parms: callable
            Function used to estimate initial parameters from dataset
    """
    return functions.get(name, None)

def names():
    """
    Return the list of names for the functions
    """
    return functions.keys()

