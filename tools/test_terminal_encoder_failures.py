"""Bounded real-process fault injection. Run only under the shared GPU lease."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--smoke-case", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tree / "pipeline"))
    import offline_toolbox as box
    args.work.mkdir(parents=True, exist_ok=False)
    reports = []
    # Use a successful, completed FFV1 case's command. Its input demux may
    # already be cleaned; reproduce that compressed input in OUR fault folder.
    original = next((args.smoke_case / "work").glob("job-*/command.json"))
    command = json.loads(original.read_text(encoding="utf-8"))
    raw = args.work / "input.h264"
    subprocess.run([str(args.runtime / "media/ffmpeg.exe"), "-v", "error", "-i", str(args.source),
                    "-map", "0:v:0", "-c:v", "copy", "-bsf:v", "h264_mp4toannexb", "-an", str(raw)], check=True)
    for flag, value in (("--input", raw), ("--report", args.work / "native-failed.json"),
                        ("--cancel-file", args.work / "native-cancel")):
        command[command.index(flag)+1] = str(value)
    env = dict(os.environ, TEMP=str(args.work), TMP=str(args.work), PYTHONIOENCODING="utf-8",
               XESS_TERMINAL_FFMPEG=str(args.runtime / "media/ffmpeg.exe"),
               XESS_TERMINAL_OUTPUT=str(args.work / "missing-parent/forbidden.mkv"),
               XESS_TERMINAL_LOG=str(args.work / "terminal.log"),
               XESS_TERMINAL_CANCEL=str(args.work / "native-cancel"))
    start = time.monotonic()
    try:
        box.run_process(command, cwd=args.runtime / "bin", environment=env,
                        log=args.work / "native.log", cancelled=lambda: False, timeout=50)
        raise AssertionError("encoder output failure was ignored")
    except box.OfflineError:
        elapsed = time.monotonic()-start
        assert elapsed < 45
        assert not (args.work / "missing-parent/forbidden.mkv").exists()
        assert "error" in (args.work / "terminal.log").read_text(errors="replace").lower()
        reports.append(dict(case="encoder_exits_before_connect", passed=True, seconds=elapsed))

    cancel = args.work / "cancel"
    request = dict(input=str(args.source), output=str(args.work / "cancelled.mkv"), backend="gpu-block",
                   mode="sr-fg", scale=1.5, depth="ai", encoder="ffv1")
    request_path = args.work / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    status = args.work / "status.json"
    # Cancel only after FFmpeg has received RGB bytes, exercising an active
    # named pipe rather than cancelling at the planning check.
    worker = [str(args.runtime / "python/python.exe"), str(args.tree / "pipeline/offline_toolbox.py"),
              "run", "--request", str(request_path), "--runtime-root", str(args.runtime),
              "--work-root", str(args.work / "cancel-work"), "--cancel-file", str(cancel), "--status-file", str(status)]
    with (args.work / "cancel.log").open("wb") as log:
        proc = subprocess.Popen(worker, stdout=log, stderr=subprocess.STDOUT, env=env)
        deadline=time.monotonic()+40
        while proc.poll() is None and time.monotonic()<deadline:
            files=list((args.work / "cancel-work").glob("job-*/output.encoded.mkv"))
            if any(f.stat().st_size > 256*1024 for f in files):
                break
            time.sleep(.02)
        assert proc.poll() is None, "job ended before active encoder cancellation"
        start=time.monotonic()
        cancel.touch()
        proc.wait(timeout=25)
    state=json.loads(status.read_text(encoding="utf-8"))
    assert state["status"]=="cancelled",state
    assert not Path(request["output"]).exists()
    reports.append(dict(case="cancel_active_software_encoder", passed=True, seconds=time.monotonic()-start))
    (args.work / "results.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
    print(json.dumps(reports),flush=True)


if __name__ == "__main__":
    main()
