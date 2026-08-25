#!/usr/bin/env python3
"""Optional per-stage timing shared by every Python SR/FG component.

Timing is disabled by default so ordinary user logs stay untouched.  Enable it
with ``--stage-timing`` on the drivers or with the ``XESS_STAGE_TIMING``
environment variable; each component then appends one machine-readable JSON
line to stderr when it finishes::

    [timing] component=prepare-sr {"decoder_read_s": .., "analyze_total_s": ..}

xess-vsr.exe understands the same environment variable and prints the matching
``component=xess-vsr`` line (CPU wall clock plus D3D12 GPU timestamp deltas).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time


TIMING_ENV = "XESS_STAGE_TIMING"


def timing_requested(flag: bool | None = None, environment=None) -> bool:
    """True when ``--stage-timing`` was passed or ``XESS_STAGE_TIMING`` is set."""
    if flag:
        return True
    environment = os.environ if environment is None else environment
    return str(environment.get(TIMING_ENV, "")).strip().lower() not in ("", "0", "false", "off")


class StageTimer:
    """Accumulates wall-clock seconds and call counts per named stage."""

    def __init__(self, on: bool | None = None):
        self.on = timing_requested(on)
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextlib.contextmanager
    def span(self, name: str):
        if not self.on:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - started)
            self.counts[name] = self.counts.get(name, 0) + 1

    def observe(self, name: str, seconds: float, calls: int = 1) -> None:
        """Fold an externally measured value (e.g. a ring's own wait counter) in."""
        if not self.on:
            return
        self.totals[name] = self.totals.get(name, 0.0) + float(seconds)
        self.counts[name] = self.counts.get(name, 0) + calls

    def report(self, component: str, stream=None) -> dict[str, float]:
        if not self.on:
            return {}
        payload = {f"{name}_s": round(seconds, 4) for name, seconds in sorted(self.totals.items())}
        payload["calls"] = {name: self.counts[name] for name in sorted(self.counts)}
        print(f"[timing] component={component} {json.dumps(payload)}",
              file=stream or sys.stderr, flush=True)
        return payload
