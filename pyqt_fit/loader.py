from __future__ import print_function, absolute_import
import inspect
from pathlib import Path
import imp
import sys
import re

bad_chars = re.compile(u'\W')


python_version = sys.version_info

if python_version.major == 2 and python_version.minor == 7:

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

elif python_version.major == 3 and python_version.minor >= 3:
    from importlib import machinery as ilm

    module_exts = ilm.all_suffixes()
    module_exts.append('.pyx')
    module_exts = module_exts[::-1]

    def create_loader(pack_name, filepath):
        ext = filepath.ext
        if ext in ilm.SOURCE_SUFFIXES:
            return ilm.SourceFileLoader(pack_name, str(filepath))
        if ext in ilm.BYTECODE_SUFFIXES:
            return ilm.SourcelessFileLoader(pack_name, str(filepath))
        if ext in ilm.EXTENSION_SUFFIXES:
            return ilm.ExtensionFileLoader(pack_name, str(filepath))

    if python_version.minor == 3:

        def create_module(loader):
            " Version for Python 3.3 "
            return loader.load_module()

    else:
        from types import ModuleType

        def create_module(loader):
            " Version for Python 3.4 or later "
            mod = ModuleType(loader.name)
            loader.exec_module(mod)
            return mod

    module_loaders = [ (ilm.EXTENSION_SUFFIXES, ilm.ExtensionFileLoader),
                       (ilm.SOURCE_SUFFIXES, ilm.SourceFileLoader),
                       (ilm.BYTECODE_SUFFIXES, ilm.SourcelessFileLoader) ]

    def load_module(pack_name, module_name, search_path):
        pth = Path(search_path) / module_name
        for exts, loader_cls in module_loaders:
            for ext in exts:
                filename = pth / Path(ext)
                if filename.exists():
                    loader = loader_cls(pack_name, str(filename))
                    mod = create_module(loader)
                    if mod is not None:
                        return mod

else:
    raise ImportError("This module can only be imported with python 2.7 and 3.x where x >= 3")

def load(find_functions, search_path=None):
    """
    Load the modules in the search_path.
    If search_path is None, then load modules in the same folder as the function looking for them.
    """
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    system_files = [caller_module.__file__]
    module_path = Path(caller_module.__file__).resolve().parent
    sys_files = set()
    for f in system_files:
        if f.endswith(".pyo") or f.endswith(".pyc"):
            f = f[:-3] + "py"
        sys_files.add(Path(f).resolve())
    if search_path is None:
        search_path = module_path
    else:
        search_path = Path(search_path).resolve()
    fcts = {}
# Search for python, cython and modules
    modules = set()
    for ext in module_exts:
        # for f in search_path.files("*" + ext):
        for f in list(search_path.glob("*" + ext)):
            if f.stem[:2] != '__':
                module_name = f.stem
                modules.add(module_name)
    for module_name in modules:
        pack_name = '%s.%s_%s' % (caller_module.__name__,
                                  bad_chars.sub('_', str(module_path)),
                                  module_name)
        try:
            mod = load_module(pack_name, module_name, search_path)
            fcts.update(find_functions(mod))
        except ImportError as ex:
            print("Warning, cannot import module '{0}' from {1}: {2}"
                  .format(module_name, caller_module.__name__, ex), file=sys.stderr)
    return fcts
