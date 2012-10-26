import inspect
from path import path
import imp
import sys

sys_modules = "*.so"
if sys.platform == 'win32' or sys.platform == 'cygwin':
    sys_modules = "*.dll"
elif sys.platform == 'darwin':
    sys_modules = "*.dylib"

def load(find_functions):
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    system_files = [ caller_module.__file__ ]
    sys_files = set()
    for f in system_files:
        if f.endswith(".pyo") or f.endswith(".pyc"):
            f = f[:-3]+"py"
        sys_files.add(path(f).abspath())
    search_path = path(caller_module.__file__).abspath().dirname()
    fcts = {}
# Search for python, cython and modules
    for f in (search_path.files("*.py") + search_path.files("*.pyx") + search_path.files(sys_modules)):
        if f not in sys_files:
            module_name = f.namebase
            pack_name = '%s.%s' % (caller_module.__name__,module_name)
            try:
                mod = sys.modules.get(pack_name)
                if mod:
                    imp.reload(mod)
                else:
                    mod_desc = imp.find_module(module_name, caller_module.__path__)
                    mod = imp.load_module(pack_name, *mod_desc)
                    globals()[module_name] = mod
                fcts.update(find_functions(mod))
            except ImportError as ex:
                print("Warning, cannot import module '%s' from %s: %s" % (module_name, caller_module.__name__, ex))
    return fcts


