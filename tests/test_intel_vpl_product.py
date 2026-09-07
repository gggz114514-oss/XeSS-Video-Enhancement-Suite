"""Failure and timeline contracts for the independent Intel media route."""
import pathlib
import sys
import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'pipeline'))
import fast_pro as f

class ProductTests(unittest.TestCase):
    def info(self):
        return f.MediaInfo('input.mp4',864,480,8,24000,1001,8*1001/24000,
                           False,1,'yuv420p',None,None,None,None,None)

    def test_combined_keeps_fraction_and_uses_one_native_stage(self):
        dec,vpp,enc=f.build_commands('sr-fg',self.info(),1296,720,
            ffmpeg='ffmpeg',vpl='vpl',source='input.mp4',partial='output.mp4')
        self.assertEqual(vpp[1],'sr-fi')
        self.assertEqual(vpp[vpp.index('--out-fps-num')+1],'48000')
        self.assertEqual(vpp[vpp.index('--out-fps-den')+1],'1001')
        self.assertEqual(f.expected_output_frames('sr-fg',8),16)
        self.assertNotIn('-frames:v',enc)  # audio tail must drain completely

    def test_vfr_that_hides_in_average_rate_is_rejected(self):
        pts=[float(Fraction(i*1001,24000)) for i in range(8)]
        pts[3]+=0.012
        with patch.object(f,'_frame_pts',return_value=pts):
            with self.assertRaisesRegex(f.FastProError,'VFR'):
                f.source_timeline('probe','ffmpeg',self.info())

    def test_cfr_rational_source_timeline(self):
        pts=[round(float(Fraction(i*1001,24000)),6) for i in range(8)]
        with patch.object(f,'_frame_pts',return_value=pts):
            self.assertEqual(f.source_timeline('probe','ffmpeg',self.info())['fps_den'],1001)

    def test_unsupported_mode_is_not_fi_fallback(self):
        with self.assertRaises(f.FastProError): f.expected_output_frames('typo',8)

    def test_portrait_two_x_does_not_require_unmeasured_excess_scale(self):
        plan=f.plan_sr_route(1080,1920,2160,3840)
        self.assertEqual(plan.route,'rotate')
        self.assertEqual(plan.detail['ai_out'],[3840,2160])

    def test_color_signal_only_on_supported_fi_extension(self):
        info=replace(self.info(),color_space='bt709',color_range='tv')
        common=dict(ffmpeg='ffmpeg',vpl='vpl',source='input.mp4',partial='output.mp4')
        _,sr,_=f.build_commands('sr-fg',info,1296,720,**common)
        _,fi,_=f.build_commands('fi',info,864,480,**common)
        self.assertNotIn('--matrix',sr)
        self.assertIn('--matrix',fi)
        self.assertIn('limited',fi)

    def test_unvalidated_full_range_sr_is_not_metadata_only_success(self):
        info=replace(self.info(),color_space='bt709',color_range='pc')
        with patch.object(f,'probe_media',return_value=info):
            with self.assertRaisesRegex(f.FastProError,'颜色信号'):
                f.run_fast_pro(mode='sr',source_path='in',output_path='out',ffprobe='probe',
                    ffmpeg='ffmpeg',vpl='vpl',work_dir='work',out_width=1296,out_height=720)

    def test_matroska_audio_tail_is_not_video_duration_or_vfr(self):
        data={'streams':[{'codec_type':'video','width':864,'height':480,
               'nb_read_frames':'8','avg_frame_rate':'24/1','r_frame_rate':'24/1',
               'pix_fmt':'yuv420p'},{'codec_type':'audio'}], 'format':{'duration':'1.0'}}
        with patch.object(f,'_probe_json',return_value=data):
            info=f.probe_media('probe','with-audio-tail.mkv')
        self.assertEqual(info.duration_s,1/3)
        self.assertFalse(info.is_vfr)

    def test_pts_tolerance_uses_stream_timebase(self):
        info=replace(self.info(),time_base='1/24000')
        pts=[float(Fraction(i*1001,24000)) for i in range(8)]
        pts[3]+=.0005
        with patch.object(f,'_frame_pts',return_value=pts):
            with self.assertRaisesRegex(f.FastProError,'VFR'):
                f.source_timeline('probe','ffmpeg',info)

if __name__=='__main__': unittest.main()
