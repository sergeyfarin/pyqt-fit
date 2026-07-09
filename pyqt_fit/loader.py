from __future__ import print_function, absolute_import
import inspect
import os
import sys
import re

bad_chars = re.compile(u'\\W')


python_version = sys.version_info

if python_version[0] == 2:

    import imp

    if sys.platform == 'win32' or sys.platform == 'cygwin':
        module_exts = ['.dll']
    elif sys.platform == 'darwin':
        module_exts = ['.dylib']
    else:
        module_exts = ['.so']
    module_exts += ['.pyx', '.pyc', '.py']

    def load_module(pack_name, module_name, search_path):
        """ Version for Python 2.7 """
        mod_desc = imp.find_module(module_name, [search_path])
        return imp.load_module(pack_name, *mod_desc)

elif python_version[0] == 3:
    from importlib import machinery as ilm
    from importlib import util
    from types import ModuleType

    module_loaders = [(ilm.EXTENSION_SUFFIXES, ilm.ExtensionFileLoader),
                      (ilm.BYTECODE_SUFFIXES, ilm.SourcelessFileLoader),
                      (ilm.SOURCE_SUFFIXES, ilm.SourceFileLoader),
                      (['.pyx'], ilm.SourceFileLoader)]
    module_exts = [ext for exts, loader_cls in module_loaders for ext in exts]

    def load_module(pack_name, module_name, search_path):
        pth = os.path.join(search_path, module_name)
        for exts, loader_cls in module_loaders:
            for ext in exts:
                filename = pth + ext
                if os.path.isfile(filename):
                    loader = loader_cls(pack_name, filename)
                    spec = util.spec_from_file_location(pack_name, filename,
                                                         loader=loader)
                    if hasattr(util, 'module_from_spec'):
                        mod = util.module_from_spec(spec)
                    else:
                        mod = ModuleType(pack_name)
                        mod.__file__ = filename
                    sys.modules[pack_name] = mod
                    spec.loader.exec_module(mod)
                    return mod
        raise ImportError("Cannot find module '{}' in {}".format(module_name,
                                                                  search_path))

else:
    raise ImportError("This module can only be imported with python 2.7 and 3.x where x >= 3")

def load(find_functions, search_path=None):
    """
    Load the modules in the search_path.
    If search_path is None, then load modules in the same folder as the function looking for them.
    """
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    module_path = os.path.dirname(os.path.abspath(caller_module.__file__))
    if search_path is None:
        search_path = module_path
    else:
        search_path = os.path.abspath(search_path)
    fcts = {}
# Search for python, cython and modules
    modules = set()
    for ext in module_exts:
        for filename in os.listdir(search_path):
            if filename.endswith(ext) and filename[:2] != '__':
                module_name = filename[:-len(ext)]
                modules.add(module_name)
    for module_name in modules:
        pack_name = '%s.%s_%s' % (caller_module.__name__,
                                  bad_chars.sub('_', module_path),
                                  module_name)
        try:
            mod = load_module(pack_name, module_name, search_path)
            fcts.update(find_functions(mod))
        except ImportError as ex:
            print("Warning, cannot import module '{0}' from {1}: {2}"
                  .format(module_name, caller_module.__name__, ex), file=sys.stderr)
    return fcts
