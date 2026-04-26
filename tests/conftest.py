"""Make `clinical_rl` (vendored inside ``house_md_env/``) importable as a bare
top-level module from the test suite, without requiring ``pip install -e .``.

This mirrors the runtime PYTHONPATH set by the Space's ``Dockerfile`` (which
adds ``/app/env/house_md_env`` to ``sys.path`` so that ``server/*.py`` can do
``from clinical_rl.env import ...`` directly).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_DIR = _REPO_ROOT / "house_md_env"
for _p in (_REPO_ROOT, _PKG_DIR):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
