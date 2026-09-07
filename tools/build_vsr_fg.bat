@echo off
rem Build the offline SR/FG worker entry points (xess-vsr.exe, xess-fg.exe)
rem from THIS worktree's sources.  The runtime stage previously shipped
rem gate9-era binaries that predate the shared-ring output contract
rem (--out-shm-name / --ui-mask / --reset-frames), so they cannot be driven by
rem the accepted pipeline.  Requirements: MSVC BuildTools (vcvars64) and the
rem XeSS SDK layout at %XESS_SDK% (default ..\..\..\sdk\official).
setlocal enabledelayedexpansion
if not defined VSCMD_PATH (echo Set VSCMD_PATH to VsDevCmd.bat & exit /b 2)
call "%VSCMD_PATH%" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 (
  echo vcvars64 failed
  exit /b 1
)
set "HERE=%~dp0"
for %%I in ("%HERE%..") do set "REPO=%%~fI"
if "%XESS_SDK%"=="" set "XESS_SDK=%REPO%\..\..\..\sdk\official"
if not exist "%XESS_SDK%\inc\xess\xess.h" (
  echo XeSS SDK not found at %XESS_SDK%
  exit /b 1
)
rem Generated executables stay out of the source worktree; default is the
rem landing run area, overridable for disposable builds.
if "%XESS_VSRFG_BUILD_DIR%"=="" set "XESS_VSRFG_BUILD_DIR=%REPO%\..\..\..\work\r4-toolbox-landing-v2\build\vsr-fg"
set "OUT=%XESS_VSRFG_BUILD_DIR%"
if not exist "%OUT%" mkdir "%OUT%"
set "LOG=%OUT%\build_vsr_fg.log"

echo =^> building xess-vsr.exe > "%LOG%"
cl /nologo /W3 /O2 /std:c++17 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS ^
   /I"%XESS_SDK%\inc" /I"%REPO%\src" ^
   /Fo:"%OUT%\xess_vsr.obj" /Fe:"%OUT%\xess-vsr.exe" ^
   "%REPO%\src\xess_vsr.cpp" ^
   /link "%XESS_SDK%\lib\libxess.lib" d3d12.lib dxgi.lib /SUBSYSTEM:CONSOLE >> "%LOG%" 2>&1
if errorlevel 1 (type "%LOG%" & echo xess-vsr build FAILED & exit /b 1)

echo =^> building xess-fg.exe >> "%LOG%"
cl /nologo /W3 /O2 /std:c++17 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS ^
   /I"%XESS_SDK%\inc" /I"%REPO%\src" ^
   /Fo:"%OUT%\xess_fg.obj" /Fe:"%OUT%\xess-fg.exe" ^
   "%REPO%\src\xess_fg.cpp" ^
   /link "%XESS_SDK%\lib\libxess_fg.lib" "%XESS_SDK%\lib\libxell.lib" "%XESS_SDK%\lib\libxess.lib" ^
   d3d12.lib dxgi.lib d3d11.lib d3dcompiler.lib windowsapp.lib user32.lib /SUBSYSTEM:CONSOLE >> "%LOG%" 2>&1
if errorlevel 1 (type "%LOG%" & echo xess-fg build FAILED & exit /b 1)

echo OK: %OUT%\xess-vsr.exe %OUT%\xess-fg.exe
exit /b 0
