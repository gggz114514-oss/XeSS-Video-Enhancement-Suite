"""The bootstrap installer must work with a stripped PowerShell module path."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class InstallerSourceTests(unittest.TestCase):
    def test_sha256_verification_uses_framework_crypto(self) -> None:
        source = (ROOT / "install_runtime.ps1").read_text(encoding="utf-8")
        executable = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertNotIn("Get-FileHash", executable)
        self.assertIn("System.Security.Cryptography.SHA256", executable)


if __name__ == "__main__":
    unittest.main()
