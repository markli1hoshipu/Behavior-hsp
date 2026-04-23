"""Delegate msgpack imports to the policy-server virtualenv package.

The Isaac Sim conda env used for eval is missing ``msgpack``. Importing the
entire policy-server site-packages tree is unsafe because it overrides unrelated
packages, so we forward only this package.
"""

from __future__ import annotations

from pathlib import Path


_REAL_DIR = Path("/home/aaron/behavior-1k-solution/.venv/lib/python3.11/site-packages/msgpack")
_REAL_INIT = _REAL_DIR / "__init__.py"

if not _REAL_INIT.exists():
    raise ModuleNotFoundError(f"Bundled msgpack fallback not found at {_REAL_INIT}")

__file__ = str(_REAL_INIT)
__path__ = [str(_REAL_DIR)]

with _REAL_INIT.open("r", encoding="utf-8") as f:
    exec(compile(f.read(), __file__, "exec"), globals(), globals())
