@echo off
setlocal
REM XeSS portable entry: run_xess.bat <input.mp4> [scale] [--quality Q] [--frames N] [--out-dir DIR]
"%~dp0python\python.exe" "%~dp0run_xess.py" %*
