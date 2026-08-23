"""Fused SEA-RAFT gather-correlate forward kernel for PyTorch XPU.

See ``loader.py`` for availability handling and the ``XESS_XPU_CORR``
environment switch, ``setup.py`` / ``build.cmd`` for compilation, and
``corr_stats`` / ``reset_corr_stats`` (``XESS_XPU_CORR_STATS=1``) for the
test-only staged/fallback counters.
"""

from .loader import (
    ENV_LIB_DIR,
    ENV_MODE,
    ENV_STATS,
    corr_stats,
    current_mode,
    gather_correlate_pyramid,
    is_available,
    load,
    reset_corr_stats,
    smoke_add,
    status_text,
)

__all__ = [
    "ENV_LIB_DIR",
    "ENV_MODE",
    "ENV_STATS",
    "corr_stats",
    "current_mode",
    "gather_correlate_pyramid",
    "is_available",
    "load",
    "reset_corr_stats",
    "smoke_add",
    "status_text",
]
