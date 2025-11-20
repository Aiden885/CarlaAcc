"""
Matlab engine factory with simple caching.
Ensures the MATLAB session is initialized once and reused by all managers.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

try:
    import matlab.engine as matlab_engine  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    matlab_engine = None

_engine_lock = threading.Lock()
_cached_engine = None


def get_matlab_engine(workdir: Optional[str] = None, force: bool = True):
    """
    Lazily start a MATLAB engine and return a cached instance.

    Args:
        workdir: Optional directory that the engine should cd into.
        force: When False, return None instead of starting a new engine.
    """
    global _cached_engine

    if matlab_engine is None and not force:
        return None

    with _engine_lock:
        if _cached_engine is None:
            if matlab_engine is None:
                if not force:
                    return None
                raise RuntimeError("MATLAB engine package is not available.")

            _cached_engine = matlab_engine.start_matlab()
            if workdir is None:
                workdir = os.getcwd()
            _cached_engine.cd(workdir)

        return _cached_engine


def shutdown_matlab_engine():
    """Close the cached MATLAB engine if one was started."""
    global _cached_engine
    with _engine_lock:
        if _cached_engine is not None:
            try:
                _cached_engine.quit()
            finally:
                _cached_engine = None
