"""nabia signal (C++ bindings)."""

from ._core import signal as _ext
from ._reexport import export as _export

__all__ = _export(_ext, globals())
__doc__ = _ext.__doc__
