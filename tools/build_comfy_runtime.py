"""Package the audited nested runtime, never a developer worktree or UI bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_manager import sha256_file
from tools.publication_policy import is_private_gpu_source

REQUIRED = (
    'bin/gpu-block-native.exe', 'bin/gpu-dis-native.exe', 'bin/amd-of-native.exe',
    'bin/xess-vsr.exe', 'bin/xess-fg.exe', 'bin/vpl-ai-vpp.exe', 'bin/libvpl.dll',
    'bin/libxess.dll', 'bin/libxess_fg.dll', 'bin/libxell.dll',
    'bin/openvino.dll', 'bin/openvino_intel_gpu_plugin.dll',
    'bin/vcruntime140.dll', 'bin/vcruntime140_1.dll', 'bin/msvcp140.dll',
    'media/ffmpeg.exe', 'probe/ffprobe.exe', 'python/python.exe',
    'models/depth-anything-v2-small/depth_anything_v2_small.xml',
    'models/depth-anything-v2-small/depth_anything_v2_small.bin',
    'shaders/common/native_sr_effects.dxil', 'shaders/common/native_depth_rgb.cso',
    'shaders/common/native_rgba_share.cso', 'shaders/common/native_rgba_ingress.dxil',
    'shaders/dis/dis_native_gray.dxil',
    'licenses/INTEL_XESS_SDK_LICENSE.txt', 'licenses/THIRD_PARTY_NOTICES.md',
)


def collect(runtime, supplement=None):
    records = {}
    for base in (Path(runtime), Path(supplement) if supplement else None):
        if base is None:
            continue
        for file in sorted(base.rglob('*')):
            if not file.is_file():
                continue
            rel = file.relative_to(base)
            if is_private_gpu_source(rel):
                raise ValueError('发布包不得包含 GPU DIS / GPU Block 源码：' + str(rel))
            if rel.parts[0] not in ('bin', 'media', 'probe', 'python', 'models', 'shaders', 'licenses'):
                continue
            if '__pycache__' in rel.parts or file.suffix.lower() in ('.pyc', '.pyo', '.log', '.pdb', '.obj', '.blob', '.partial'):
                continue
            if rel.parts[0] == 'models' and rel.parts[1] != 'depth-anything-v2-small':
                continue
            if file.is_symlink() or file.name.lower().startswith(('obs-', 'obs64', 'libobs')) or file.suffix.lower() in ('.mp4', '.raw', '.avi'):
                raise ValueError('发布包包含非运行时内容：' + str(rel))
            records[rel.as_posix()] = file
    missing = set(REQUIRED) - records.keys()
    if missing:
        raise ValueError('运行时缺少必需文件：' + ', '.join(sorted(missing)))
    return records


def build(runtime, output, version, supplement=None, source_version='1.4.1'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    asset_name = f'xess-comfy-runtime-windows-x64-{version}.zip'
    archive = output / asset_name
    if archive.exists():
        raise FileExistsError('不覆盖封存发布包：' + str(archive))
    records = collect(runtime, supplement)
    file_hashes = {n: sha256_file(f) for n, f in records.items()}
    size = sum(f.stat().st_size for f in records.values())
    partial = archive.with_suffix('.zip.partial')
    try:
        with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name, file in records.items():
                z.write(file, 'xess-comfy-runtime/' + name)
        partial.rename(archive)
    finally:
        partial.unlink(missing_ok=True)
    tag = 'runtime-' + version
    manifest = dict(schema_version=2, source_version=source_version, layout='comfy-r4-nested-v1',
                    runtime_version=version, release_tag=tag, release_status='not-published',
                    asset_name=asset_name, archive_root='xess-comfy-runtime',
                    download_url=f'https://github.com/gggz114514-oss/XeSS-Video-Enhancement-Suite/releases/download/{tag}/{asset_name}',
                    sha256=sha256_file(archive), archive_size=archive.stat().st_size,
                    installed_size=size, required_files=list(REQUIRED), file_hashes=file_hashes)
    (output / 'runtime_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output / (asset_name+'.sha256')).write_text(manifest['sha256']+'  '+asset_name+'\n', encoding='ascii')
    print(json.dumps({k: v for k,v in manifest.items() if k not in ('file_hashes','required_files')}, ensure_ascii=False))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--supplement')
    p.add_argument('--version', default='2026.09.08-r4.1')
    p.add_argument('--source-version', default='1.4.1')
    a = p.parse_args()
    build(a.runtime, a.output, a.version, a.supplement, a.source_version)
