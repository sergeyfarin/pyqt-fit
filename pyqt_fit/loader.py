from __future__ import print_function, absolute_import
import inspect
from path import path
import imp
import sys
import re

bad_chars = re.compile(u'\W')

sys_modules = "*.so"
if sys.platform == 'win32' or sys.platform == 'cygwin':
    sys_modules = "*.dll"
elif sys.platform == 'darwin':
    sys_modules = "*.dylib"


def load(find_functions, search_path=None):
    """
    Load the modules in the search_path.
    If search_path is None, then load modules in the same folder as the function looking for them.
    """
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    system_files = [caller_module.__file__]
    module_path = path(caller_module.__file__).abspath().dirname()
    sys_files = set()
    for f in system_files:
        if f.endswith(".pyo") or f.endswith(".pyc"):
            f = f[:-3] + "py"
        sys_files.add(path(f).abspath())
    if search_path is None:
        search_path = module_path
    else:
        search_path = path(search_path).abspath()
    fcts = {}
# Search for python, cython and modules
    for f in (search_path.files("*.py") +
              search_path.files("*.pyx") +
              search_path.files(sys_modules)):
        if f not in sys_files:
            module_name = f.namebase
            pack_name = '%s.%s_%s' % (caller_module.__name__,
                                      bad_chars.sub('_', module_path),
                                      module_name)
            try:
                mod_desc = imp.find_module(module_name, [search_path])
                mod = imp.load_module(pack_name, *mod_desc)
                fcts.update(find_functions(mod))
            except ImportError as ex:
                print("Warning, cannot import module '{0}' from {1}: {2}"
                      .format(module_name, caller_module.__name__, ex), file=sys.stderr)
    return fcts
