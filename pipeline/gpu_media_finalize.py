"""Shared native compressed-output finalizer; no processing pixels on the CPU."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def probe(ffprobe, path):
    return json.loads(subprocess.check_output([
        str(ffprobe), '-v', 'error', '-show_streams', '-show_packets',
        '-show_data_hash', 'sha256', '-show_entries',
        'packet=stream_index,pts,dts,duration,data_hash', '-of', 'json', str(path)], timeout=300))


def stream_packets(data, kind):
    stream = next((s for s in data['streams'] if s['codec_type'] == kind), None)
    return stream, [] if stream is None else [p for p in data['packets'] if p['stream_index'] == stream['index']]


def source_contract(ffprobe, source):
    data = probe(ffprobe, source)
    if sum(s['codec_type'] == 'audio' for s in data['streams']) > 1:
        raise ValueError('此 finalizer 当前只验证单音轨，不能把未验证的多音轨标为通过')
    video, packets = stream_packets(data, 'video')
    if not video or not packets or any('pts' not in p for p in packets):
        raise ValueError('输入没有可验证的视频 PTS')
    pts = sorted(int(p['pts']) for p in packets)
    tb = Fraction(video['time_base'])
    if pts[0] != 0:
        raise ValueError('此 finalizer 尚不支持非零原始 PTS 起点；必须由共用 wrapper 明确处理，不能静默归零')
    if len(pts) > 1:
        step = pts[1] - pts[0]
        if step <= 0 or any(b-a != step for a, b in zip(pts, pts[1:])):
            raise ValueError('此 native 入口为 CFR；VFR/重复 PTS 必须明确拒绝，不能静默重采样')
        fps = 1 / (step * tb)
    else:
        fps = Fraction(video['r_frame_rate'])
    if fps <= 0:
        raise ValueError('无法确定正的有理数帧率')
    return data, video, pts, fps


def decoded_hashes(ffmpeg, path):
    result = subprocess.run([str(ffmpeg), '-v', 'error', '-i', str(path), '-map', '0:v:0',
                             '-fps_mode', 'passthrough', '-f', 'framemd5', '-'], capture_output=True, timeout=300)
    if result.returncode or result.stderr.strip():
        raise RuntimeError('完整视频软件解码校验失败：' + result.stderr.decode(errors='replace'))
    return [line.split(',')[-1].strip() for line in result.stdout.decode().splitlines()
            if line and not line.startswith('#')]


def finalize(*, source, elementary, native_report, provider_report, output, ffmpeg, ffprobe, report,
             backend='amd-of', cancelled=lambda: False):
    source, elementary, output = map(Path, (source, elementary, output))
    native = json.loads(Path(native_report).read_text(encoding='utf-8'))
    provider = json.loads(Path(provider_report).read_text(encoding='utf-8')) if provider_report else {}
    if (native.get('gate_status') not in ('PASS_GPU_EXECUTION', 'PASS_COLOR_MOTION_DEPTH_PENDING')
            or native['pipeline']['motion_backend'] != backend):
        raise ValueError('原生报告不是所请求的已完成 GPU 结果，禁止以替代后端成片冒充成功')
    data, video, original_pts, fps = source_contract(ffprobe, source)
    n = len(original_pts)
    mode = native['pipeline']['mode']
    expected = n if mode == 'sr' else 2*n-1
    count = native['shared_core_counters']
    if (native['frames']['input'] != n or native['frames']['output'] != expected
            or count['source_pair_analysis_count'] != n-1 or count['motion_after_sr_count'] != 0
            or (backend == 'amd-of' and (provider.get('sdk_directional_call_count') != 3*n-2
                or provider.get('source_pair_analysis_count') != n-1))):
        raise ValueError('完整原片映射/共享分析计数不匹配，不能发布截断片')
    partial = output.with_name(output.stem + '.partial' + output.suffix)
    if output.exists() or partial.exists():
        raise FileExistsError('不覆盖已有成片/待诊断 partial')
    output.parent.mkdir(parents=True, exist_ok=True)
    rate = fps if mode == 'sr' else fps*2
    audio, source_audio_packets = stream_packets(data, 'audio')
    # Give elementary H264 a real container time base first. Combining its
    # missing DTS with -shortest and a second input can starve the audio input
    # on FFmpeg 7.1. Both steps are compressed stream copies, never re-encoding.
    timed = output.with_name(output.stem + '.timed.partial' + output.suffix)
    if timed.exists():
        raise FileExistsError('不覆盖此前保留的定时码流中间容器')
    settings = native['encode_settings']
    software = settings.get('encoder') in ('libx264', 'libx265', 'ffv1')
    # All supported endpoints use B=0 and one packet per frame. Reconstruct
    # exact CFR timestamps without decoding/re-encoding: MKV millisecond PTS
    # otherwise leak into MP4 and HEVC parser delay can discard initial frames.
    retime = (f'setts=pts=N*{rate.denominator}:dts=N*{rate.denominator}:'
              f'duration={rate.denominator}:time_base=1/{rate.numerator}')
    timing_command = [str(ffmpeg), '-v', 'error', *([] if software else ['-r', str(rate)]), '-i', str(elementary),
                      '-map', '0:v:0', '-c', 'copy', '-bsf:v', retime, '-avoid_negative_ts', 'disabled', str(timed)]
    subprocess.run(timing_command, capture_output=True, check=True, timeout=300)
    command = [str(ffmpeg), '-v', 'error', '-i', str(timed), '-i', str(source),
               '-map', '0:v:0', '-map', '1:a?', '-c', 'copy', '-t', f'{float(Fraction(expected)/rate):.12f}',
               '-avoid_negative_ts', 'disabled', '-movflags', '+faststart', str(partial)]
    subprocess.run(command, capture_output=True, check=True, timeout=300)
    output_data = probe(ffprobe, partial)
    out_video, video_packets = stream_packets(output_data, 'video')
    out_audio, audio_packets = stream_packets(output_data, 'audio')
    expected_codec = {'h264_qsv': 'h264', 'hevc_qsv': 'hevc', 'libx264': 'h264',
                      'libx265': 'hevc', 'ffv1': 'ffv1'}[settings.get('encoder', 'h264_qsv')]
    if out_video.get('codec_name') != expected_codec:
        raise RuntimeError('实际编码格式与用户选择不一致，未发布成片')
    out_tb = Fraction(out_video['time_base'])
    pts = sorted(int(p['pts']) for p in video_packets)
    pts_gate = len(pts) == expected and all(abs(p*out_tb-Fraction(i)/rate) <= out_tb for i, p in enumerate(pts))
    reference = decoded_hashes(ffmpeg, elementary)
    actual = decoded_hashes(ffmpeg, partial)
    pixels_gate = len(actual) == expected and reference == actual
    audio_gate = (not audio and not out_audio) or (bool(out_audio) and bool(audio_packets)
        and audio['codec_name'] == out_audio['codec_name']
        and [p['data_hash'] for p in audio_packets] == [p['data_hash'] for p in source_audio_packets[:len(audio_packets)]])
    geometry = native['dimensions']['final_output']
    geometry_gate = [out_video['width'], out_video['height']] == [geometry['width'], geometry['height']]
    lossless_gate = None
    if settings.get('encoder') == 'ffv1':
        hashed = subprocess.run([str(ffmpeg), '-v', 'error', '-i', str(partial), '-map', '0:v:0',
            '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', '-f', 'hash', '-hash', 'sha256', '-'],
            capture_output=True, check=True, timeout=300)
        actual_rgb = hashed.stdout.decode().strip().split('=')[-1].lower()
        lossless_gate = bool(settings.get('rgb_sha256')) and actual_rgb == settings['rgb_sha256']
        if not lossless_gate:
            raise RuntimeError('FFV1 无损像素哈希不一致，未发布成片')
    if not (pts_gate and pixels_gate and audio_gate and geometry_gate):
        raise RuntimeError(f'封装校验失败：pts={pts_gate},pixels={pixels_gate},audio={audio_gate},geometry={geometry_gate}; '
                           f'decoded={len(reference)}/{len(actual)}，保留 partial，不发布成片')
    result = {'schema_version': 'offline-route-media@1', 'backend': backend, 'operation': mode,
              'classification': 'EXPERIMENTAL_ONLY' if backend != 'gpu-block' else 'VALIDATED_CORE', 'status': 'validated_pending_publication',
              'requested': {'input': str(source), 'output': str(output)},
              'planned': {'input_frames': n, 'output_frames': expected, 'fps': str(rate), 'audio': 'copy' if audio else 'none'},
              'applied': {'output_frames': len(actual), 'geometry': [out_video['width'], out_video['height']],
                          'fps': str(rate), 'time_base': str(out_tb), 'pts_first': pts[0], 'pts_last': pts[-1],
                          'audio_packets': len(audio_packets), 'encoder': native['encode_settings'],
                          'terminal_post': native['terminal_post']},
              'source_sha256': sha(source), 'elementary_sha256': sha(elementary), 'output_sha256': sha(partial),
              'native_report': str(native_report), 'native_report_sha256': sha(native_report),
              'provider_report': str(provider_report), 'provider_report_sha256': sha(provider_report) if provider_report else None,
              'commands': [timing_command, command], 'timed_bitstream_container': str(timed),
              'shared_core_counters': count, 'boundaries': native['boundaries'],
              'pts_gate': pts_gate, 'pixels_unchanged_by_mux': pixels_gate,
              'ffv1_rgb_lossless_gate': lossless_gate,
              'audio_payload_prefix_exact': audio_gate, 'geometry_gate': geometry_gate,
              'validation': '仅最终压缩成片的软件全片解码哈希，不读取原生 GPU 中间纹理。',
              'pts_policy': '原片已验证零起点CFR；重建原有理数网格。FG2N-1不添加末尾外推帧。',
              'audio_policy': '原音轨压缩payload前缀逐包SHA256一致；按视频终点裁剪较长音轨，不因较短音轨丢视频帧。'}
    # Write metadata before making the finished video visible. Failed report I/O
    # leaves only the explicitly partial video, never an unindexed final file.
    Path(report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if cancelled():
        raise RuntimeError('任务取消，未发布成片，保留本次 partial 供检查')
    # Cross-process publish must not overwrite a result from another job.
    # Windows rename is exclusive; POSIX use link + unlink for the same effect.
    if os.name == 'nt':
        partial.rename(output)
    else:
        os.link(partial, output)
        partial.unlink()
    result['status'] = 'complete'
    Path(report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    timed.unlink(missing_ok=True)
    return result
