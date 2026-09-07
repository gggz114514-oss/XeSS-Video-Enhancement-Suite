"""Read-only installation diagnostic. Does not require Comfy torch or render."""
import json
import subprocess
import sys
from pathlib import Path
from runtime_manager import load_manifest, default_engine, engine_compatible, RuntimeManagerError

def main():
    try:
        m=load_manifest()
        root=default_engine(m)
        if not engine_compatible(root,m,full=True):
            raise RuntimeManagerError('运行时完整校验未通过，请执行 install_runtime.bat -Force')
        for binary in (root/'media/ffmpeg.exe',root/'probe/ffprobe.exe'):
            result=subprocess.run([str(binary),'-version'],capture_output=True,timeout=15)
            if result.returncode: raise RuntimeManagerError('无法启动：'+str(binary))
        result=subprocess.run([str(root/'python/python.exe'),str(Path(__file__).parent/'pipeline/offline_toolbox.py'),
                               'capabilities','--runtime-root',str(root)],capture_output=True,timeout=30)
        if result.returncode: raise RuntimeManagerError(result.stderr.decode('utf-8',errors='replace'))
        print(result.stdout.decode('utf-8'))
        return 0
    except (RuntimeManagerError,OSError,subprocess.SubprocessError) as exc:
        print('[XeSS R4] '+str(exc),file=sys.stderr)
        return 1

if __name__=='__main__': raise SystemExit(main())
