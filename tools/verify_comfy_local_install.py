"""Real Comfy HTTP prompt test; run against an isolated server under GPU lease."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8197')
    p.add_argument('--input', default='r4_demo_480p.mp4')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output-directory', type=Path, required=True)
    p.add_argument('--ffprobe', required=True)
    p.add_argument('--frames', type=int, default=8)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    def call(route, value=None):
        req = Request(args.url + route, data=None if value is None else json.dumps(value).encode(),
                      headers={'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=45) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from exc
    names = ['XeSSR4OfflineSuperResolution', 'XeSSR4OfflineFrameGeneration',
             'XeSSR4OfflineSuperResolutionFrameGeneration']
    info = {name: call('/object_info/' + name)[name] for name in names}
    for name, node in info.items():
        assert node['output'] == ['VIDEO', 'STRING'], (name, node['output'])
        assert len(node['input']['required']['encoder'][0]) == 6
    all_nodes = call('/object_info')
    assert not any(name.startswith('XeSSVideo') for name in all_nodes), 'Legacy R3 nodes remain registered'
    (args.work / 'node-info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    reports = []
    for mode, name in enumerate(names):
        inputs = {'video': ['1', 0], 'backend': 'GPU Block（快速）', 'encoder': '自动',
                  'depth': 'AI 深度', 'sharpen': True}
        if mode != 1:
            inputs.update(scale='1.5×', custom_scale=1.5, five_frame=True, anti_stripe=True)
        prompt = {
        '1': {'class_type': 'LoadVideo', 'inputs': {'file': args.input}},
        '2': {'class_type': name, 'inputs': inputs},
        '3': {'class_type': 'SaveVideo', 'inputs': {
            'video': ['2', 0], 'filename_prefix': f'video/R4_install_{mode}', 'format': 'auto', 'codec': 'auto'}}}
        (args.work / f'prompt{mode}.json').write_text(json.dumps(prompt, ensure_ascii=False, indent=2), encoding='utf-8')
        queued = call('/prompt', {'prompt': prompt})
        assert not queued.get('node_errors'), queued
        prompt_id = queued['prompt_id']
        print('Queued ' + name + ' ' + prompt_id, flush=True)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            history = call('/history/' + prompt_id)
            if prompt_id in history:
                result = history[prompt_id]
                (args.work / f'history{mode}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                assert result['status']['status_str'] == 'success', result['status']
                break
            time.sleep(1)
        else:
            raise TimeoutError('Comfy prompt did not finish within 240 seconds')
        files = list(args.output_directory.glob(f'video/R4_install_{mode}*.mp4'))
        assert len(files) == 1, files
        streams = json.loads(subprocess.check_output([args.ffprobe, '-v', 'error', '-count_frames',
            '-show_entries', 'stream=codec_type,codec_name,width,height,nb_read_frames,avg_frame_rate',
            '-of', 'json', str(files[0])]))['streams']
        video = next(s for s in streams if s['codec_type'] == 'video')
        frames = args.frames if mode == 0 else args.frames * 2 - 1
        size = (864, 480) if mode == 1 else (1296, 720)
        assert int(video['nb_read_frames']) == frames and (video['width'], video['height']) == size, streams
        assert any(s['codec_type'] == 'audio' for s in streams), streams
        reports.append({'node': name, 'prompt_id': prompt_id, 'output': str(files[0]), 'streams': streams})
    report = {'ok': True, 'nodes_registered': names, 'old_node_still_registered': False, 'cases': reports}
    (args.work / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
