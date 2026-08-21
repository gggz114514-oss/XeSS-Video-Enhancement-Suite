@echo off
setlocal
set "ROOT=%~dp0"
set "RUNTIME_ROOT=%COMFYUI_XESS_RUNTIME%"
if not defined RUNTIME_ROOT set "RUNTIME_ROOT=%ROOT%.runtime"
if /I "%RUNTIME_ROOT:~-7%"=="\engine" (set "ENGINE=%RUNTIME_ROOT%") else (set "ENGINE=%RUNTIME_ROOT%\engine")
if not exist "%ENGINE%\python\python.exe" call "%ROOT%install_runtime.bat"
if errorlevel 1 exit /b 1
"%ENGINE%\python\python.exe" "%ROOT%runtime_manager.py" ensure
if errorlevel 1 exit /b 1
"%ENGINE%\python\python.exe" "%ENGINE%\run_xess.py" %*
exit /b %errorlevel%
