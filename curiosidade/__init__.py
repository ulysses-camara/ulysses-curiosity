# pylint: disable='missing-module-docstring'
from .core import ProbingModelContainer, attach_probers
from .probers import ProbingModelWrapper, ProbingModelFactory
from .probers.tasks.tasks import *

try:
    import importlib.metadata as _importlib_metadata

except ModuleNotFoundError:
    import _importlib_metadata  # type: ignore


try:
    __version__ = _importlib_metadata.version(__name__)

except _importlib_metadata.PackageNotFoundError:
    __version__ = "0.1.0"


__all__ = [
    "core",
    "probers",
]
