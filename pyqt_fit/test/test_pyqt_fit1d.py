from __future__ import division, absolute_import, print_function

import pytest


try:
    from .. import pyqt_fit1d
except ImportError:
    pytest.skip("PySide6 or PySide2 is required for the GUI tests", allow_module_level=True)


def test_gui_module_imports():
    assert pyqt_fit1d is not None

