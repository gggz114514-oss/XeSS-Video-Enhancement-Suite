"""Arc A workaround: optional node field -> plan -> compiled worker flag."""
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'pipeline'))
import comfy_offline_nodes as nodes
import offline_toolbox as box


class ArcACompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.mp4'
        self.source.write_bytes(b'test fixture')
        self.request = dict(input=str(self.source), output=str(self.root / 'out.mp4'))

    def command(self, mode, enabled=None, backend='gpu-block'):
        r = dict(self.request, mode=mode, backend=backend)
        if enabled is not None:
            r['arc_a_compat'] = enabled
        r = box.validate_request(r)
        p = dict(applied=r, codec='h264', source_frames=12, source_fps='24',
                 output_width=1296, output_height=720)
        return box.native_command(p, box.runtime_paths(self.root), self.root, 'cancel')

    def test_omitted_switch_preserves_default_command(self):
        for mode in box.MODES:
            self.assertEqual(self.command(mode), self.command(mode, False))
            self.assertNotIn('--decode-share', self.command(mode))

    def test_all_three_modes_forward_compatibility_once(self):
        for mode in box.MODES:
            cmd = self.command(mode, True)
            self.assertEqual(cmd.count('--decode-share'), 1)
            self.assertEqual(cmd[cmd.index('--decode-share') + 1], 'rgba-d3d11')
            self.assertEqual(cmd[cmd.index('--motion-repair') + 1], 'refine')
            self.assertEqual(cmd[cmd.index('--gpu-anti-stripe') + 1], 'off')

    def test_other_backends_explicitly_rejected(self):
        for backend in ('cpu-dis', 'intel-vpl-ai'):
            with self.subTest(backend=backend), self.assertRaisesRegex(box.OfflineError, '仅支持 GPU Block'):
                box.validate_request(dict(self.request, backend=backend, arc_a_compat=True))

    def test_all_shared_gpu_routes_forward_all_modes(self):
        for backend in ('gpu-block', 'gpu-dis', 'amd-of'):
            for mode in box.MODES:
                with self.subTest(backend=backend, mode=mode):
                    command = self.command(mode, True, backend)
                    self.assertEqual(command[command.index('--decode-share')+1], 'rgba-d3d11')
                    self.assertEqual(command[command.index('--motion-backend')+1], backend)
                    self.assertNotIn('--decode-share', self.command(mode, False, backend))

    def test_boolean_validation_not_truthy_string(self):
        for value in ('false', 'true', 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(box.OfflineError, '布尔'):
                box.validate_request(dict(self.request, arc_a_compat=value))

    def test_missing_shaders_rejected_before_media_probe(self):
        with patch.object(box, 'missing_runtime', return_value=[]):
            with self.assertRaisesRegex(box.OfflineError, 'native_rgba_share.cso'):
                box.plan(dict(self.request, arc_a_compat=True), self.root)

    def test_dis_integer_gray_shader_is_checked_before_native_launch(self):
        paths = box.runtime_paths(self.root)
        missing = box.missing_runtime(paths, 'gpu-dis')
        self.assertIn(str(paths['dis_shaders'] / 'dis_native_rgba_gray.dxil'), missing)
        for backend in ('gpu-block', 'amd-of', 'cpu-dis', 'intel-vpl-ai'):
            self.assertNotIn(str(paths['dis_shaders'] / 'dis_native_rgba_gray.dxil'),
                             box.missing_runtime(paths, backend))

    def test_optional_widget_and_old_api_defaults_all_nodes(self):
        with patch.object(nodes, '_capabilities', return_value=box.capabilities()):
            for cls in nodes.NODE_CLASS_MAPPINGS.values():
                inputs = cls.INPUT_TYPES()
                self.assertNotIn('arc_a_compat', inputs['required'])
                self.assertFalse(inputs['optional']['arc_a_compat'][1]['default'])
                sig = inspect.signature(getattr(cls, cls.FUNCTION))
                self.assertIs(sig.parameters['arc_a_compat'].default, False)
                self.assertIs(cls.VALIDATE_INPUTS(backend='GPU Block（快速）', arc_a_compat=True), True)
                self.assertIn('仅支持', cls.VALIDATE_INPUTS(backend='CPU DIS（稳定）', arc_a_compat=True))

    def test_all_nodes_execute_forward_checkbox(self):
        for cls in nodes.NODE_CLASS_MAPPINGS.values():
            node = cls()
            with patch.object(node, '_execute', return_value=('video', 'out')) as run:
                getattr(node, cls.FUNCTION)(video='fixture', arc_a_compat=True)
                self.assertIs(run.call_args.kwargs['arc_a_compat'], True)

    def test_request_forwarding_and_default(self):
        args = dict(mode='sr', source=self.source, output=self.root/'out.mp4',
                    backend='gpu-block', scale=1.5, encoder='auto', depth='ai',
                    sharpen=False, five_frame=False, anti_stripe=False)
        self.assertFalse(nodes._request(**args)['arc_a_compat'])
        self.assertTrue(nodes._request(**args, arc_a_compat=True)['arc_a_compat'])


if __name__ == '__main__':
    unittest.main()
