"""SR fused post-processor tests.

Covers fixed/adaptive sharpening, the vertical ringing-guard blend, odd
dimensions, degenerate inputs, and the failure paths of the full
``sr_postprocess`` process (upstream input ends early, guide decoder ends
early, stdout closes early, threads exit within a bounded time).

Golden hashes below were generated once on the reference stack
(Python 3.13.11, numpy 2.3.5, OpenCV 5.0.0) from the fused implementation;
they lock the byte-exact output of the retired chain's replacement.  A fresh
run of the same code must reproduce them bit for bit.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import cv2  # noqa: F401
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "pipeline"))

import sr_postprocess as sp  # noqa: E402

W, H = 17, 13  # odd dimensions exercise the scalar tails
FRAMES = 4
IN_W, IN_H = 21, 15


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fresh_rng() -> np.random.Generator:
    """Golden hashes were generated from one generator with a fixed seed.

    The fixed-cases draw 8 frames, the adaptive cases then draw 24 more
    (draws 8..31), and the guard cases draw 8 sources last (draws 32..39).
    Tests below replay that exact draw order.
    """
    return np.random.default_rng(20260826)


def draw_frames(rng, count: int) -> list[np.ndarray]:
    return [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(count)]


def draw_sources(rng, count: int) -> list[np.ndarray]:
    return [rng.integers(0, 256, (H * 3, W * 3, 3), dtype=np.uint8)
            for _ in range(count)]


class SharpenAlgorithmTests(unittest.TestCase):
    """In-process byte-exact and invariant checks for the fused sharpen pass."""

    def test_fixed_sharpen_matches_golden(self) -> None:
        golden = [
            "45fd29a117566ac1f5b8cc6a2295d51f343cb27add571ef5a70a6ff2500f2e84",
            "4cf8342dfb5ed3506b48e7d372a0a0a9a8f1b5ae96a138966d828541668ac1a5",
            "7362a65bbafc53b1b3064ed68292c36643b2c0582010416a2b7e3f36e6ffa563",
            "d16077a657e2ab4b269729876a7e0ae1f70ccdab75d09ca4b327ff3313c77536",
            "864717bca37e7cb7852d53845173db11d5e6036f9085297145912ce266077f4c",
            "165472f3a296e1d122fed7d0cd4a8d89acbbde457e8df97996e867c06b6a0bf4",
            "c7b25a917af88f57a5a62537db53d6d2e711b229adf907239e48d01c36ca3cf4",
            "5006404225ba5addb6a3737192285668c41be4f8802078d556908e41416cedba",
        ]
        buf = sp._SharpenBuffers(H, W)
        frames = draw_frames(_fresh_rng(), 8)
        for case, expected in enumerate(golden):
            out = sp.sharpen_frame(buf, frames[case], "fixed", 0.35, 0.18)
            self.assertEqual(sha256_bytes(out.copy().tobytes()), expected,
                             f"fixed case {case} drifted")

    def test_adaptive_sequences_match_golden(self) -> None:
        golden = [
            ["ed993f634722d9209c57856e01316434abafcd5c95d0f33bff8614dd80fad25e",
             "c55ae863ceb51f2accdb6ed127340424dcd262ba8e8b017209bfaf2386d020f1",
             "d50a6c6b6b342b75a2f1d28af975fe18ed903e8d93c32ad5143a5b4272c460e0"],
            ["0ce901d31a2094a7ed170acbd9c11d097d941fc477bc3eedb616feec41b2d003",
             "41a2f48ef18b74f875d460d7da61ed735e76b4d805d9449162363be2d8a3cc55",
             "ebf31cfa39736871c27cce18959e35f003f2a9a44b7f5ff6ea7d9712ef0c15e4"],
            ["89c88e9bba0bafa15b28f0d0b56aaf21382034f18b69e63c062b7c813e3fc381",
             "8ef10bb3e599d39d2659abe420135cf89f70b95df278ced29b57ea76d34e2160",
             "05e9878f4e9f567c2eb4c277eb41ca0b5d3fc754255204089164d9255c58f0b7"],
            ["440b5cc020050dace6849fc5741f91ba1196a012a0fa84c1c224eb7d8a729a44",
             "fae73fd3f1f0c273eaef0c40c24a58f6ba841f0cd1c9a4e00f7553620ce4ff14",
             "d64222e8da90c1f7d596f5435ac6d355f477d34c33eadb9b835b075f250c51bb"],
            ["31ca77df415240a6f14de5be9c697b864f958b15133f9d239876ea3db894c18b",
             "86a7b7a599b07495d0dd8bae280fa2be02cd844f2fbaf4618a653ae24179929b",
             "664ba319666dad29ba4d7b4cf3084ec88b87f787bf6a8ad709cef3b4fbc2fb0c"],
            ["24fe96303ea0feb2c213364ea59ff397e516f86c2f36bd946bd12ce0ce0c58a0",
             "1e6e6e88714c5f5421416735b37ade9fd84e456c9aff628dfb16f4b13ab5410d",
             "c5b10db7cb7efd64b9ce32d61c94272a0d13abb9b2b662b9e26cc47b2f50cd34"],
            ["46b47f8c06045716cf0b4fe1f6c3e1a82cb9030a224f2e2c95028f088d474557",
             "b48e2d4db002c3aa0b018ccbe44dc295578ba33be52a061504e5776f2238edc4",
             "53dfc1e03ad595feda0e4238046e14ea6ae03a481bb89cef8c21138d2106c2e6"],
            ["1c954275a84a9494b3d9649f18a267f17601da305f8560b434ea9c9d26f47f62",
             "525179b8af8729df364b6c41b140b1c0a3dd816faac23a17b957a1cb24b65f49",
             "b5e1814fd6ec5852e1df41ecab0b2c4bd72ec8975a422f6d6da43a5ead3f2ce8"],
        ]
        rng = _fresh_rng()
        draw_frames(rng, 8)  # replay: skip the fixed-case draws
        for case, expected in enumerate(golden):
            buf = sp._SharpenBuffers(H, W)
            frames = draw_frames(rng, 3)
            for index, frame in enumerate(frames):
                out = sp.sharpen_frame(buf, frame, "adaptive", 0.35, 0.18)
                self.assertEqual(sha256_bytes(out.copy().tobytes()), expected[index],
                                 f"adaptive case {case} frame {index} drifted")

    def test_first_adaptive_frame_equals_fixed_on_same_frame(self) -> None:
        # The first adaptive frame has zero motion, so its linear combination
        # reduces to the exact fixed-path math.
        frame = draw_frames(_fresh_rng(), 1)[0]
        fixed = sp.sharpen_frame(sp._SharpenBuffers(H, W), frame, "fixed", 0.35, 0.18)
        adaptive = sp.sharpen_frame(
            sp._SharpenBuffers(H, W), frame, "adaptive", 0.35, 0.18)
        self.assertTrue(np.array_equal(fixed, adaptive))

    def test_guard_blend_matches_golden_and_bounds(self) -> None:
        golden = [
            ("1134bdf165005253579ef1f47ef3ac5281f7523a011db1361b6dace8066d4e7a",
             "49ba3cb328342eeb7079dbc830e023d2886676fd01df5372b28c39a9b090b352"),
            ("b7860fd21c330309effd9f16fbd72e747b8b217056d384cefdc06d4e8a18792b",
             "eae8856394725d13ef47285256fa9412276e5be6d567344cc1a5ce4ade7882ac"),
            ("3ce76b44593de58534cb96e1f5279eef0be7d807e3494ce941f4580601e5d65e",
             "1f6999c54528ac0f3ec1277477bee209f5f3297dcb0ee642f4336fa7924fac1f"),
            ("b406ae6c1cf2c3c53854bcf9e52954cb16141f4add96033562de86bf1e4e662e",
             "e464b2f1a6cdadb347a4e38a7e7c8c75abe9679ea2bc6565237289b37c727a91"),
            ("5427a2fb8656d2f1cd8e1a71c0176829ab4d58570b26cd49f0b8878015d1b2ea",
             "fa2fcb6ec6e90c96f285afc9a823c4616c44e91fbf2eb769c4105f3fd70b691a"),
            ("cc81aabb7bca0aa8658e235eb9c4adcdbcad3ae5da87e10d04a164e020e8175c",
             "3b4ed3e71519e813c40a5be31343060322764e39d27e193566b39daa924f6103"),
            ("ff60f7cda1520861905e1437df2f03827be98e03ff23c2dcd512369faf37d94f",
             "3a3d93d70783cf194754c79275d5a3aef02bba4658ae943dec1c8802ec09b64d"),
            ("92040de8f07ea55385fd50631aef3fe864f0af3c9c9c784a1d98225db9451400",
             "a647790fae40e439c5f5504434071f444ffb32c0aff31d51b4e75e0604455900"),
        ]
        rng = _fresh_rng()
        draw_frames(rng, 32)  # replay: skip the fixed + adaptive draws
        for case, (blend_sha, inv_sha) in enumerate(golden):
            gb = sp._GuideBuffers(H, W)
            source = draw_sources(rng, 1)[0]
            sp.compute_blend(gb, source, W, H, 0.75)
            self.assertEqual(sha256_bytes(gb.blend.tobytes()), blend_sha,
                             f"guard case {case} blend drifted")
            self.assertEqual(sha256_bytes(gb.inv_blend.tobytes()), inv_sha,
                             f"guard case {case} inv_blend drifted")
            self.assertTrue(np.all(gb.blend >= 0.0) and np.all(gb.blend <= 0.90))
            self.assertTrue(np.array_equal(gb.inv_blend, 1.0 - gb.blend))

    def test_flat_frames_are_unchanged(self) -> None:
        flat = np.full((H, W, 3), 128, np.uint8)
        buf = sp._SharpenBuffers(H, W)
        self.assertTrue(np.array_equal(
            sp.sharpen_frame(buf, flat, "fixed", 0.35, 0.18), flat))
        flat_source = np.full((H * 3, W * 3, 3), 90, np.uint8)
        gb = sp._GuideBuffers(H, W)
        sp.compute_blend(gb, flat_source, W, H, 0.75)
        self.assertTrue(np.array_equal(gb.blend, np.zeros((H, W), np.float32)))
        self.assertTrue(np.array_equal(gb.inv_blend, np.ones((H, W), np.float32)))

    def test_zero_strength_is_identity(self) -> None:
        frame = draw_frames(_fresh_rng(), 1)[0]
        buf = sp._SharpenBuffers(H, W)
        out = sp.sharpen_frame(buf, frame, "fixed", 0.0, 0.0)
        self.assertTrue(np.array_equal(out, frame))

    def test_odd_dimensions_all_black_white_random(self) -> None:
        rng = _fresh_rng()
        for hw in ((25, 33), (1, 1)):
            h, w = hw
            for frame in (np.zeros((h, w, 3), np.uint8),
                          np.full((h, w, 3), 255, np.uint8),
                          rng.integers(0, 256, (h, w, 3), dtype=np.uint8)):
                buf = sp._SharpenBuffers(h, w)
                first = sp.sharpen_frame(buf, frame, "adaptive", 0.35, 0.18)
                second = sp.sharpen_frame(buf, frame, "adaptive", 0.35, 0.18)
                self.assertEqual(first.shape, (h, w, 3))
                self.assertTrue(np.array_equal(first, second))  # same frame, deterministic

    def test_output_u8_bounded_for_random_inputs(self) -> None:
        # 8 random fixed cases: values must stay in range and be deterministic.
        frames = draw_frames(_fresh_rng(), 8)
        for case, frame in enumerate(frames):
            buf = sp._SharpenBuffers(H, W)
            out = sp.sharpen_frame(buf, frame, "fixed", 1.0, 1.0).copy()
            self.assertEqual(out.dtype, np.uint8)
            again = sp.sharpen_frame(buf, frame, "fixed", 1.0, 1.0)
            self.assertTrue(np.array_equal(out, again))


class PostProcessProcessTests(unittest.TestCase):
    """End-to-end checks of ``sr_postprocess`` as a subprocess."""

    POST = os.fspath(ROOT / "pipeline" / "sr_postprocess.py")

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls._shim = os.path.join(cls._tmp.name, "fake_ffmpeg.py")
        with open(cls._shim, "w", encoding="utf-8") as handle:
            handle.write(
                "import os, sys\n"
                "args = sys.argv[1:]\n"
                "frames, w, h = 4, 1, 1\n"
                "i = 0\n"
                "while i < len(args):\n"
                "    if args[i] == '-vframes': frames = int(args[i + 1]); i += 2\n"
                "    elif args[i] == '-s': w, h = (int(x) for x in args[i + 1].split('x')); i += 2\n"
                "    else: i += 1\n"
                "out = sys.stdout.buffer\n"
                "import hashlib\n"
                "for idx in range(frames):\n"
                "    if os.environ.get('FAKE_FFMPEG_FAIL') == '1' and idx >= 1:\n"
                "        print('boom: corrupt stream', file=sys.stderr, flush=True)\n"
                "        sys.exit(1)\n"
                "    seed = hashlib.sha256(f'{idx}'.encode()).digest()\n"
                "    out.write(b''.join(bytes([seed[j % 32]]) for j in range(w * h * 3)))\n"
            )
        cls.fake_ffmpeg = f"{sys.executable} {cls._shim}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _input_frames(self, count: int = FRAMES) -> bytes:
        frame_bytes = IN_W * IN_H * 3
        return bytes([(idx * 17 + j) % 256 for idx in range(count)
                      for j in range(frame_bytes)])

    def _run(self, extra, input_data, *, gc_fail=False, timeout=60):
        env = os.environ.copy()
        if gc_fail:
            env["FAKE_FFMPEG_FAIL"] = "1"
        command = [sys.executable, self.POST, "--width", str(W), "--height", str(H),
                   "--frames", str(FRAMES), *extra]
        return subprocess.run(command, input=input_data, capture_output=True,
                              env=env, timeout=timeout)

    def test_e2e_fixed_matches_golden(self) -> None:
        result = self._run(["--sharpen-mode", "fixed", "--static", "0.35",
                            "--motion", "0.18"], self._input_frames())
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertEqual(sha256_bytes(result.stdout),
                         "e02bcedef7f7e8cae96b89ef7d15e7344cba9abb6ded4355c4f3d42e28ca67f1")

    def test_e2e_guard_matches_golden_and_exits_cleanly(self) -> None:
        result = self._run(["--sharpen-mode", "fixed", "--static", "0.35",
                            "--motion", "0.18", "--guard-strength", "0.75",
                            "--video", "x", "--ffmpeg", self.fake_ffmpeg,
                            "--in-w", str(IN_W), "--in-h", str(IN_H)],
                           self._input_frames())
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertEqual(sha256_bytes(result.stdout),
                         "29c7a7a8a329b2f451aa4fc246c40326e84f924b0cb6730619af75e69c5200ce")

    def test_input_ends_early_exits_nonzero_without_hang(self) -> None:
        result = self._run(["--sharpen-mode", "fixed", "--static", "0.35",
                            "--motion", "0.18"],
                           self._input_frames(1), timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input ended at frame 1",
                      result.stderr.decode("utf-8", "replace"))

    def test_guide_decoder_ends_early_reports_stderr(self) -> None:
        result = self._run(["--sharpen-mode", "fixed", "--static", "0.35",
                            "--motion", "0.18", "--guard-strength", "0.75",
                            "--video", "x", "--ffmpeg", self.fake_ffmpeg,
                            "--in-w", str(IN_W), "--in-h", str(IN_H)],
                           self._input_frames(), gc_fail=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        stderr = result.stderr.decode("utf-8", "replace")
        self.assertIn("boom: corrupt stream", stderr)

    def test_stdout_closed_early_exits_without_hang(self) -> None:
        command = [sys.executable, self.POST, "--width", str(W), "--height", str(H),
                   "--frames", str(FRAMES), "--sharpen-mode", "fixed",
                   "--static", "0.35", "--motion", "0.18"]
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Close the read end immediately: the child's first stdout write must
        # fail and the process must still exit promptly.
        process.stdout.close()
        try:
            try:
                process.stdin.write(self._input_frames())
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass  # the child may exit before we finish feeding it
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            self.fail("sr_postprocess hung after stdout closed")
        stderr = process.stderr.read().decode("utf-8", "replace")
        process.stderr.close()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("OSError", stderr)  # BrokenPipeError or Errno 22, per platform


if __name__ == "__main__":
    unittest.main()