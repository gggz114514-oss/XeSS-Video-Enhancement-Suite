"""Benchmark and stability harness for the fused XPU gather-correlate kernel.

Compares FusedXPUCorrBlock against StreamingCorrBlock the way SEA-RAFT uses
them at high resolution: one correlation query per iteration, four iterations
per frame pair.

Usage (from an environment with icx NOT required - only the built binary):
    python tools/bench_xpu_corr.py [--height 1088 --width 1920]
        [--batch 1 2] [--pairs 5] [--stability]

Timings are taken between torch.xpu.synchronize() boundaries after warmup;
the first (cold) invocation is reported separately because generic spir64
JIT compilation happens once on first launch.  Memory numbers come from the
PyTorch XPU caching allocator plus a device-level sanity check via
torch.xpu.mem_get_info where available.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "pipeline" / "sea_raft_core"))

import torch  # noqa: E402

from corr import StreamingCorrBlock, resolve_highres_block_cls  # noqa: E402
from utils.utils import coords_grid  # noqa: E402

ARGS = SimpleNamespace(corr_levels=4, corr_radius=4)
ITERS_PER_PAIR = 4


def make_case(batch: int, height: int, width: int, channels: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    fmap1 = torch.randn(batch, channels, height, width,
                        generator=generator).to("xpu")
    fmap2 = torch.randn(batch, channels, height, width,
                        generator=generator).to("xpu")
    dilation = torch.ones(batch, 1, height, width, device="xpu")
    return fmap1, fmap2, dilation


def time_pairs(block, batch, height, width, pairs, seed0):
    """Return per-pair latency seconds; one pair = ITERS_PER_PAIR queries."""
    latencies = []
    for pair in range(pairs):
        gen = torch.Generator().manual_seed(seed0 + pair)
        coords = coords_grid(batch, height, width, "xpu") + \
            torch.randn(batch, 2, height, width, generator=gen,
                        device="cpu").to("xpu") * 20.0
        torch.xpu.synchronize()
        start = time.perf_counter()
        for _ in range(ITERS_PER_PAIR):
            out = block(coords)
        torch.xpu.synchronize()
        latencies.append(time.perf_counter() - start)
    return latencies


def peak_extra_bytes(block, batch, height, width):
    coords = coords_grid(batch, height, width, "xpu")
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()
    base = torch.xpu.memory_allocated()
    block(coords)
    torch.xpu.synchronize()
    return torch.xpu.max_memory_allocated() - base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=136,
                        help="feature height (native/8), e.g. 136 for 1080p")
    parser.add_argument("--width", type=int, default=240,
                        help="feature width (native/8), e.g. 240 for 1080p")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="interleave TILE x WG configs within one "
                             "process so thermal drift hits all equally")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--stability", action="store_true",
                        help="run 100 consecutive pairs and watch memory")
    args = parser.parse_args()

    if not torch.xpu.is_available():
        print("torch.xpu unavailable", file=sys.stderr)
        return 2
    from xpu_corr import loader

    if not loader.is_available():
        print(f"fused extension unavailable: {loader.status_text()}",
              file=sys.stderr)
        return 2

    device_name = torch.xpu.get_device_name(torch.xpu.current_device())
    print(f"device={device_name} dtype=fp32 "
          f"feature={args.width}x{args.height} channels={args.channels}")

    for batch in args.batch:
        fmap1, fmap2, dilation = make_case(batch, args.height, args.width,
                                           args.channels, 1234)
        fused_cls = resolve_highres_block_cls(fmap1)
        streaming = None if args.skip_streaming else \
            StreamingCorrBlock(fmap1, fmap2, ARGS)
        blocks = {}
        for tile, wg in ((4, 128), (2, 128), (8, 128), (4, 256)):
            os.environ["XESS_XPU_CORR_TILE"] = str(tile)
            os.environ["XESS_XPU_CORR_WG"] = str(wg)
            blocks[(tile, wg)] = fused_cls(fmap1, fmap2, ARGS)
        # Cold start of the default config includes first-launch JIT cost.
        warm_coords = coords_grid(batch, args.height, args.width, "xpu")
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        _ = blocks[(4, 128)](warm_coords)
        torch.xpu.synchronize()
        cold = time.perf_counter() - t0
        for block in blocks.values():
            if block is not blocks[(4, 128)]:
                _ = block(warm_coords)
        if streaming is not None:
            _ = streaming(warm_coords)
        del warm_coords

        if args.sweep:
            fused_samples = {k: [] for k in blocks}
            stream_samples = []
            for round_index in range(args.rounds):
                for key, block in blocks.items():
                    fused_samples[key] += time_pairs(
                        block, batch, args.height, args.width,
                        max(1, args.pairs // 2), 7000 + round_index)
                if streaming is not None:
                    stream_samples += time_pairs(
                        streaming, batch, args.height, args.width,
                        max(1, args.pairs // 2), 7000 + round_index)
            stream_med = (statistics.median(stream_samples)
                          if stream_samples else float("nan"))
            for key in blocks:
                med = statistics.median(fused_samples[key])
                ratio = (f" speedup={stream_med / med:.2f}x"
                         if stream_samples else "")
                print(f"B={batch} tile={key[0]} wg={key[1]} "
                      f"median/pair={med * 1e3:.1f}ms{ratio}")
            if stream_samples:
                print(f"B={batch} streaming median/pair="
                      f"{stream_med * 1e3:.1f}ms")
            continue

        os.environ["XESS_XPU_CORR_TILE"] = "4"
        os.environ["XESS_XPU_CORR_WG"] = "128"
        fused = blocks[(4, 128)]

        fused_latency = time_pairs(fused, batch, args.height, args.width,
                                   args.pairs, 5000)
        fused_med = statistics.median(fused_latency)

        row = (f"B={batch} fused[{fused_cls.__name__}] "
               f"cold={cold * 1e3:.0f}ms "
               f"median/pair={fused_med * 1e3:.1f}ms")

        if not args.skip_streaming:
            streaming = StreamingCorrBlock(fmap1, fmap2, ARGS)
            stream_latency = time_pairs(streaming, batch, args.height,
                                        args.width, args.pairs, 5000)
            stream_med = statistics.median(stream_latency)
            row += (f" streaming median/pair={stream_med * 1e3:.1f}ms "
                    f"speedup={stream_med / fused_med:.2f}x")

        fused_peak = peak_extra_bytes(fused, batch, args.height, args.width)
        row += f" fused_peak_tmp={fused_peak / 1024 ** 2:.1f}MiB"
        if not args.skip_streaming:
            stream_peak = peak_extra_bytes(StreamingCorrBlock(
                fmap1, fmap2, ARGS), batch, args.height, args.width)
            row += (f" streaming_peak_tmp={stream_peak / 1024 ** 2:.1f}MiB "
                    f"delta={(fused_peak - stream_peak) / 1024 ** 2:+.1f}MiB")
        print(row)

    if args.stability:
        batch = args.batch[0]
        fmap1, fmap2, _ = make_case(batch, args.height, args.width,
                                    args.channels, 1234)
        block = resolve_highres_block_cls(fmap1)(fmap1, fmap2, ARGS)
        torch.xpu.synchronize()
        mem_start = torch.xpu.memory_allocated()
        peak_seen = mem_start
        t0 = time.perf_counter()
        for pair in range(100):
            gen = torch.Generator().manual_seed(9000 + pair)
            coords = coords_grid(batch, args.height, args.width, "xpu") + \
                torch.randn(batch, 2, args.height, args.width,
                            generator=gen, device="cpu").to("xpu") * 25.0
            out = block(coords)
            torch.xpu.synchronize()
            peak_seen = max(peak_seen, torch.xpu.memory_allocated())
        elapsed = time.perf_counter() - t0
        mem_end = torch.xpu.memory_allocated()
        grow = (mem_end - mem_start) / 1024 ** 2
        print(f"stability: 100 pairs B={batch} {args.width}x{args.height} "
              f"in {elapsed:.1f}s | allocator growth={grow:+.1f}MiB "
              f"(start {mem_start / 1024 ** 2:.0f}, "
              f"end {mem_end / 1024 ** 2:.0f}, "
              f"peak_seen {peak_seen / 1024 ** 2:.0f} MiB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
