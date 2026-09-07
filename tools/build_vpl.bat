@echo off
rem Build the oneVPL AI VPP experiment binaries from THIS worktree's sources.
rem Requirements: MSVC BuildTools (vcvars64), oneVPL sources checked out at
rem %VPL_SRC% (default ..\..\..\tmp\libvpl-src) and a built libvpl at
rem %VPL_LIBDIR% (default ..\..\..\tmp\libvpl-build\Release).
setlocal enabledelayedexpansion
if not defined VSCMD_VER (
  if not defined VSDEVCMD (
    for /f "usebackq tokens=*" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDEVCMD=%%I\Common7\Tools\VsDevCmd.bat"
  )
  if not defined VSDEVCMD (echo Set VSDEVCMD to VsDevCmd.bat & exit /b 1)
  call "!VSDEVCMD!" -arch=x64 -host_arch=x64 >nul
  if errorlevel 1 (echo VsDevCmd failed & exit /b 1)
)
set "HERE=%~dp0"
for %%I in ("%HERE%..") do set "REPO=%%~fI"
if "%VPL_SRC%"==""    set "VPL_SRC=%REPO%\..\..\..\tmp\libvpl-src"
if "%VPL_LIBDIR%"=="" set "VPL_LIBDIR=%REPO%\..\..\..\tmp\libvpl-build\Release"
rem Keep generated executables and DLLs out of the source worktree.  The
rem caller may override this for a disposable build, but the default is the
rem E: work area used by the Fast Pro experiments.
if "%XESS_VPL_BUILD_DIR%"=="" (echo Set XESS_VPL_BUILD_DIR to an isolated build directory & exit /b 1)
set "OUT=%XESS_VPL_BUILD_DIR%"
if not exist "%OUT%" mkdir "%OUT%"
set "LOG=%OUT%\build_vpl.log"

echo =^> building vpl-ai-probe.exe > "%LOG%"
cl /nologo /W4 /O2 /std:c++17 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS ^
   /I"%VPL_SRC%\api\vpl" /I"%VPL_SRC%\api" ^
   /Fo:"%OUT%\vpl_ai_probe.obj" /Fe:"%OUT%\vpl-ai-probe.exe" ^
   "%REPO%\src\realtime\vpl_ai_probe.cpp" ^
   /link "%VPL_LIBDIR%\vpl.lib" /SUBSYSTEM:CONSOLE >> "%LOG%" 2>&1
if errorlevel 1 (type "%LOG%" & echo probe build FAILED & exit /b 1)

echo =^> building vpl-ai-vpp.exe >> "%LOG%"
cl /nologo /W4 /O2 /std:c++17 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS ^
   /I"%VPL_SRC%\api\vpl" /I"%VPL_SRC%\api" ^
   /Fo:"%OUT%\vpl_ai_vpp.obj" /Fe:"%OUT%\vpl-ai-vpp.exe" ^
   "%REPO%\src\realtime\vpl_ai_vpp.cpp" ^
   /link "%VPL_LIBDIR%\vpl.lib" /SUBSYSTEM:CONSOLE >> "%LOG%" 2>&1
if errorlevel 1 (type "%LOG%" & echo vpp build FAILED & exit /b 1)

copy /Y "%VPL_LIBDIR%\libvpl.dll" "%OUT%" >nul

echo =^> hashes >> "%LOG%"
certutil -hashfile "%OUT%\vpl-ai-probe.exe" SHA256 | findstr /v ":" >> "%LOG%"
certutil -hashfile "%OUT%\vpl-ai-vpp.exe" SHA256 | findstr /v ":" >> "%LOG%"
type "%LOG%"
echo BUILD OK; log: %LOG%
endlocal
