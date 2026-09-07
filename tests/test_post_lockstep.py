"""A producer withholding frame n+1 must receive frame n without a wait cycle."""
import concurrent.futures
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).parents[1]


class PostLockstepTests(unittest.TestCase):
    def test_no_wait_for_next_input_and_bitexact_output(self):
        command=[sys.executable,str(ROOT/'pipeline/sr_postprocess.py'),
            '--width','17','--height','13','--frames','4',
            '--sharpen-mode','off','--flush-each-frame','--threads','4']
        frames=[bytes([31+i])* (17*13*3) for i in range(4)]
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        with subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
            try:
                for frame in frames:
                    proc.stdin.write(frame);proc.stdin.flush()
                    received=pool.submit(proc.stdout.read,len(frame)).result(timeout=10)
                    self.assertEqual(received,frame)
                proc.stdin.close()
                self.assertEqual(proc.wait(timeout=10),0)
            finally:
                if proc.poll() is None:
                    proc.kill();proc.wait(timeout=10)
                pool.shutdown(wait=True)

    def test_cpu_combo_guard_uses_lockstep(self):
        source=(ROOT/'pipeline/run_pipeline.py').read_text(encoding='utf-8')
        helper=source.split('def pre_fg_guard_command',1)[1].split('def terminate_all',1)[0]
        self.assertIn('--flush-each-frame',helper)
