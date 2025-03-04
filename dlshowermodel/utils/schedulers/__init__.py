import os
import importlib
import pkgutil

# Import all modules to trigger registration
__all__ = []
for loader, module_name, is_pkg in pkgutil.iter_modules([os.path.dirname(__file__)]):
    __all__.append(module_name)
    _module = importlib.import_module(f"{__name__}.{module_name}")