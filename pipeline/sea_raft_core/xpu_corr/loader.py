"""Best-effort loader for the fused XPU gather-correlate extension.

The extension binary (``xess_xpu_corr.pyd``) is normally built in-place next
to this file or deployed to a directory pointed at by
``XESS_XPU_CORR_LIB_DIR``.  Users never compile on the fly: when the binary
is missing or fails to load we report the exact reason once and let callers
fall back to :class:`StreamingCorrBlock`.

Selection is controlled by ``XESS_XPU_CORR``:

* ``auto``      (default) load quietly, fall back on any failure,
* ``off``       never load the extension,
* ``required``  raise loudly when the extension cannot be used (testing).

No code here ever falls back to a CPU implementation: the registered op only
implements the XPU dispatch key, so wrong-device calls fail immediately.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading

log = logging.getLogger("xess.xpu_corr")

ENV_MODE = "XESS_XPU_CORR"
ENV_LIB_DIR = "XESS_XPU_CORR_LIB_DIR"
_MODULE_NAME = "xess_xpu_corr"
_VALID_MODES = ("auto", "off", "required")

_lock = threading.Lock()
_state: dict | None = None


def _mode() -> str:
    raw = os.environ.get(ENV_MODE, "auto").strip().lower()
    if raw not in _VALID_MODES:
        log.warning("%s=%r is not one of %s; treating as 'auto'.",
                    ENV_MODE, raw, ", ".join(_VALID_MODES))
        return "auto"
    return raw


def _candidate_dirs() -> list[str]:
    dirs: list[str] = []
    override = os.environ.get(ENV_LIB_DIR)
    if override:
        dirs.append(os.path.abspath(override))
    dirs.append(os.path.dirname(os.path.abspath(__file__)))
    return dirs


def _import_extension():
    # Insert lowest-priority first so XESS_XPU_CORR_LIB_DIR ends up ahead of
    # this file's directory on sys.path; otherwise the in-tree binary would
    # always shadow the override directory.
    for directory in reversed(_candidate_dirs()):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    return importlib.import_module(_MODULE_NAME)


def current_mode() -> str:
    """Effective ``XESS_XPU_CORR`` mode (validated); for selection logic."""
    return _mode()


def load() -> bool:
    """Import the extension exactly once; return True when it is usable.

    Usable means: the module imports, both ops register under the XPU dispatch
    key, an XPU device is present, and a tiny on-device probe produces correct
    results on the current PyTorch stream.
    """
    global _state
    with _lock:
        if _state is not None:
            return _state["available"]

        mode = _mode()
        if mode == "off":
            _state = {"available": False,
                      "reason": f"{ENV_MODE}=off disables the fused kernel"}
            return False

        try:
            import torch

            _import_extension()
            ops = torch.ops.xess_xpu  # AttributeError if registration failed
            ops.gather_correlate_pyramid
            ops.smoke_add

            if not torch.xpu.is_available():
                _state = {"available": False,
                          "reason": "extension loaded but torch.xpu is "
                                    "unavailable"}
                return False

            # One-off init-time probe on PyTorch's current stream; this is the
            # only synchronised call in the whole hot path's lifetime.
            probe = torch.zeros(4, device="xpu")
            result = ops.smoke_add(probe)
            torch.xpu.synchronize()
            if not bool((result == 1.0).all()):
                raise RuntimeError("smoke_add produced incorrect results")
            del probe, result

            device_name = torch.xpu.get_device_name(torch.xpu.current_device())
            _state = {"available": True, "reason": f"device={device_name}"}
            log.info("[SEA-RAFT] correlation=fused-sycl device=%s dtype=fp32",
                     device_name)
            return True
        except Exception as exc:  # noqa: BLE001 - any failure means fallback
            reason = f"{type(exc).__name__}: {exc}"
            _state = {"available": False, "reason": reason}
            if mode == "required":
                raise RuntimeError(
                    f"[SEA-RAFT] {ENV_MODE}=required but the fused XPU "
                    f"correlation extension failed to load ({reason})") from exc
            log.info("[SEA-RAFT] fused XPU correlation unavailable (%s); "
                     "using StreamingCorrBlock", reason)
            return False


def is_available() -> bool:
    return load()


def status_text() -> str:
    if _state is None:
        return "not probed yet"
    return _state["reason"]


def gather_correlate_pyramid(fmap1, fmap2_levels, coords, dilation,
                             level_scales, radius):
    """Run the fused kernel; raises RuntimeError with the load reason if the
    extension is unusable instead of silently degrading."""
    import torch

    if not load():
        raise RuntimeError(
            "[SEA-RAFT] fused XPU correlation is not usable: "
            f"{status_text()}")
    return torch.ops.xess_xpu.gather_correlate_pyramid(
        fmap1, list(fmap2_levels), coords, dilation, list(level_scales),
        int(radius))


def smoke_add(input_tensor):
    """Run the trivial +1 probe kernel (stream/dispatch sanity checks)."""
    import torch

    if not load():
        raise RuntimeError(
            "[SEA-RAFT] fused XPU correlation is not usable: "
            f"{status_text()}")
    return torch.ops.xess_xpu.smoke_add(input_tensor)


ENV_STATS = "XESS_XPU_CORR_STATS"


def _stats_module():
    import torch  # noqa: F401 - ensures libtorch is up before the pyd loads

    if not load():
        raise RuntimeError(
            "[SEA-RAFT] fused XPU correlation is not usable: "
            f"{status_text()}")
    return importlib.import_module(_MODULE_NAME)


def corr_stats():
    """Return cumulative [pure_staged, any_fallback] work-group counts.

    Counters only advance while ``XESS_XPU_CORR_STATS=1`` is set for the
    calls being measured; ``reset_corr_stats()`` zeroes them.  Test/bench
    observability only - the release hot path submits stat-free kernels.
    """
    return list(_stats_module().corr_stats())


def reset_corr_stats() -> None:
    """Zero the staged/fallback counters (no-op unless stats are enabled)."""
    _stats_module().reset_corr_stats()
