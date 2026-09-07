"""Offline fault injection: real ZIP, real disk activation, OS lock, no GPU."""
import contextlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile
import runtime_manager as r


class UpgradeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / '.runtime'
        self.base.mkdir()
        self.asset = self.root / 'runtime.zip'
        self.files = {'bin/worker.exe': b'worker-binary', 'python/python.exe': b'python-binary'}
        with zipfile.ZipFile(self.asset, 'w') as z:
            for name, data in self.files.items():
                z.writestr('rt/'+name, data)
        import hashlib
        self.m = dict(schema_version=2, layout='comfy-r4-nested-v1', runtime_version='test-r4',
                      asset_name='runtime.zip', archive_root='rt', release_status='published',
                      sha256=r.sha256_file(self.asset), archive_size=self.asset.stat().st_size,
                      installed_size=sum(map(len, self.files.values())), required_files=list(self.files),
                      file_hashes={n:hashlib.sha256(d).hexdigest() for n,d in self.files.items()},
                      download_url='https://github.com/gggz114514-oss/XeSS-Video-Enhancement-Suite/releases/download/test/runtime.zip')
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps(self.m))
        for p in (patch.object(r,'MANIFEST_PATH',self.manifest),
                  patch.dict(os.environ, {r.RUNTIME_ENV:str(self.base), r.ASSET_ENV:'',
                                          'COMFYUI_XESS_SKIP_RUNTIME_DOWNLOAD':'0'})):
            p.start()
            self.addCleanup(p.stop)

    def test_fresh_install_full_hash_and_idempotence(self):
        engine = r.ensure_runtime(asset=str(self.asset))
        self.assertTrue(r.engine_compatible(engine, full=True))
        with patch.object(r,'_extract_archive',side_effect=AssertionError('no reinstall')):
            self.assertEqual(r.ensure_runtime(asset=str(self.asset)), engine)
        self.assertFalse(list(self.base.glob('installing-*')))

    def test_r3_engine_is_preserved(self):
        old = self.base/'engine'/'python'
        old.mkdir(parents=True)
        (old/'python.exe').write_bytes(b'R3 unchanged')
        engine = r.ensure_runtime(asset=str(self.asset))
        self.assertNotEqual(engine.parent, old.parent)
        self.assertEqual((old/'python.exe').read_bytes(),b'R3 unchanged')

    def test_bad_zip_hash_does_not_touch_previous(self):
        engine = r.ensure_runtime(asset=str(self.asset))
        bad = self.root/'bad.zip'
        bad.write_bytes(b'invalid')
        (engine/'bin/worker.exe').write_bytes(b'needs-repair')
        with self.assertRaisesRegex(r.RuntimeManagerError,'SHA256'):
            r.ensure_runtime(force=True, asset=str(bad))
        self.assertEqual((engine/'bin/worker.exe').read_bytes(),b'needs-repair')

    def test_modified_file_not_hidden_by_state(self):
        engine = r.ensure_runtime(asset=str(self.asset))
        (engine/'bin/worker.exe').write_bytes(b'changed')
        self.assertFalse(r.engine_compatible(engine))
        r.ensure_runtime(asset=str(self.asset))
        self.assertTrue(r.engine_compatible(engine,full=True))
        self.assertEqual(len(list(self.base.glob('repair-backup-*'))),1)

    def test_no_space_does_not_download(self):
        with patch.object(r.shutil,'disk_usage',return_value=type('Usage',(),{'free':1})()), \
             patch.object(r,'_download') as download, self.assertRaisesRegex(r.RuntimeManagerError,'空间不足'):
            r.ensure_runtime()
        download.assert_not_called()

    def test_unpublished_has_no_network_but_local_candidate_can_be_tested(self):
        self.m['release_status']='not-published'
        self.manifest.write_text(json.dumps(self.m))
        with patch.object(r,'_download') as dl, self.assertRaisesRegex(r.RuntimeManagerError,'尚未发布'):
            r.ensure_runtime()
        dl.assert_not_called()
        self.assertTrue(r.ensure_runtime(asset=str(self.asset)).is_dir())

    def test_unsafe_paths(self):
        for name in ('../a','/a','C:/a','a:b','a/../b','a\\b','a/NUL','a/trailing.','a//b'):
            with self.subTest(name=name), self.assertRaises(r.RuntimeManagerError):
                r._relative(name)

    def test_extra_or_traversal_or_duplicate_entries_are_rejected(self):
        for name in ('rt/../escape','rt/bin/unlisted.exe','rt/bin/worker.exe'):
            archive = self.root/'unsafe.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('rt/bin/worker.exe',b'worker-binary')
                if name == 'rt/bin/worker.exe':
                    # Case-insensitive duplicate on Windows.
                    name = 'rt/BIN/worker.exe'
                z.writestr(name,b'bad')
            with self.assertRaises(r.RuntimeManagerError):
                r._extract_archive(archive,self.base,self.m)
            self.assertFalse(list(self.base.glob('installing-*')))

    def test_symlink_rejected(self):
        archive=self.root/'link.zip'
        with zipfile.ZipFile(archive,'w') as z:
            i=zipfile.ZipInfo('rt/bin/worker.exe')
            i.external_attr=(stat.S_IFLNK|0o777)<<16
            z.writestr(i,'../../escape')
        with self.assertRaisesRegex(r.RuntimeManagerError,'符号链接'):
            r._extract_archive(archive,self.base,self.m)

    def test_two_concurrent_installers_extract_once(self):
        answers, errors = [], []
        original = r._extract_archive
        with patch.object(r,'_extract_archive',wraps=original) as extract:
            def worker():
                try: answers.append(r.ensure_runtime(asset=str(self.asset)))
                except Exception as e: errors.append(e)
            workers=[threading.Thread(target=worker) for _ in range(2)]
            for w in workers: w.start()
            for w in workers: w.join(5)
            self.assertFalse(any(w.is_alive() for w in workers))
            self.assertFalse(errors)
            self.assertEqual(len(answers),2)
            self.assertEqual(extract.call_count,1)

    def test_lock_timeout_is_bounded_and_releases(self):
        with r._install_lock(self.base):
            with self.assertRaisesRegex(r.RuntimeManagerError,'超时'):
                with r._install_lock(self.base,timeout=.02): pass
        with r._install_lock(self.base,timeout=.02): pass

    def test_truncated_download_is_removed(self):
        response=io.BytesIO(b'partial')
        with patch.object(r.urllib.request,'urlopen',return_value=response), \
             self.assertRaisesRegex(r.RuntimeManagerError,'提前结束'):
            r._download(self.m['download_url'],self.root/'out.zip',100)
        self.assertFalse((self.root/'out.zip.partial').exists())

    def test_network_failure_preserves_old_version(self):
        old=self.base/'engine'
        old.mkdir()
        (old/'keep').write_bytes(b'R3')
        with patch.object(r,'_download',side_effect=OSError('network unavailable')), \
             self.assertRaisesRegex(r.RuntimeManagerError,'安装失败'):
            r.ensure_runtime()
        self.assertEqual((old/'keep').read_bytes(),b'R3')

    def test_manifest_rejects_missing_hash_and_cross_project_url(self):
        for update in ({'sha256':''},{'file_hashes':{}},{'download_url':'https://example.com/runtime.zip'}):
            self.manifest.write_text(json.dumps({**self.m,**update}))
            with self.assertRaises(r.RuntimeManagerError): r.load_manifest()


class NodeReleaseTests(unittest.TestCase):
    def test_only_three_nodes_registered_and_bootstrap_wired(self):
        import ast
        root=Path(__file__).resolve().parents[1]
        text=(root/'__init__.py').read_text()
        self.assertNotIn('xess_nodes', text)
        self.assertIn('start_background_update()',text)
        import comfy_offline_nodes as n
        self.assertEqual(len(n.NODE_CLASS_MAPPINGS),3)
        self.assertEqual(n.XeSSR4OfflineFrameGeneration.VALIDATE_INPUTS(
            backend='GPU Block（快速）', depth='固定深度'),
            '此 GPU 路线需要 AI 深度，请选择「AI 深度」。')


if __name__ == '__main__': unittest.main()
