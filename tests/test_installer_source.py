"""Manual bootstrap delegates to the same verified Python installer as nodes."""
from pathlib import Path
import unittest

class InstallerSourceTests(unittest.TestCase):
    def test_one_installer_no_pip_or_host_environment_mutation(self):
        text = (Path(__file__).resolve().parents[1] / 'install_runtime.ps1').read_text()
        self.assertIn('runtime_manager.py', text)
        self.assertIn('python_embeded', text)
        self.assertNotIn('pip install', text)
        self.assertNotIn('SetEnvironmentVariable', text)
        self.assertNotIn('Expand-Archive', text)
