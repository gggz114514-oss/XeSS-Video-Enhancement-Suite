from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))

import xess_nodes  # noqa: E402


class XpuEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        xess_nodes._XPU_PROBE.clear()
        self.engine = ROOT
        self.python = os.path.abspath(sys.executable)

    def test_probe_matches_runtime_import_order_and_allocates(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "XPU_PROBE test True 1\n", "")
        with mock.patch.object(xess_nodes.subprocess, "run", return_value=completed) as run:
            ok, _ = xess_nodes._xpu_probe(self.python, self.engine, os.environ.copy())
        self.assertTrue(ok)
        script = run.call_args.args[0][2]
        self.assertLess(script.index("import torch"), script.index("import openvino"))
        self.assertIn("torch.zeros(1, device='xpu')", script)
        self.assertIn("torch.xpu.synchronize()", script)

    def test_failed_node_environment_restores_launcher_environment(self) -> None:
        node_env = {"PATH": "same-path", "TEMP": "E:\\xess-work"}
        launcher = {
            "PATH": "same-path",
            "SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS": "1",
            "TEMP": "D:\\launcher-temp",
        }
        outcomes = iter(((False, "node failed"), (True, "launcher passed")))
        with mock.patch.dict(os.environ, launcher, clear=True), \
                mock.patch.object(xess_nodes, "_xpu_probe", side_effect=outcomes):
            selected = xess_nodes._xpu_python(self.engine, "sea-raft", node_env)
        self.assertEqual(selected, self.python)
        self.assertEqual(node_env["PATH"], "same-path")
        self.assertEqual(node_env["SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS"], "1")
        self.assertEqual(node_env["TEMP"], "E:\\xess-work")
        self.assertEqual(xess_nodes._XPU_PROBE[self.python], "inherited-main")

    def test_level_zero_selector_is_last_resort_not_default(self) -> None:
        node_env = {"PATH": "node-path", "TEMP": "E:\\xess-work"}
        launcher = {"PATH": "launcher-path", "SYCL_DEVICE_FILTER": "bad-filter"}
        outcomes = iter((
            (False, "node failed"),
            (False, "launcher failed"),
            (True, "selector passed"),
        ))
        with mock.patch.dict(os.environ, launcher, clear=True), \
                mock.patch.object(xess_nodes, "_xpu_probe", side_effect=outcomes):
            xess_nodes._xpu_python(self.engine, "sea-raft", node_env)
        self.assertEqual(node_env["ONEAPI_DEVICE_SELECTOR"], "level_zero:*")
        self.assertNotIn("SYCL_DEVICE_FILTER", node_env)
        self.assertEqual(xess_nodes._XPU_PROBE[self.python], "level-zero")

    def test_dis_uses_portable_python_without_probe(self) -> None:
        env = {"PATH": "unchanged"}
        with mock.patch.object(xess_nodes, "_xpu_probe") as probe:
            selected = xess_nodes._xpu_python(self.engine, "dis", env)
        self.assertEqual(selected, os.fspath(self.engine / "python" / "python.exe"))
        probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
