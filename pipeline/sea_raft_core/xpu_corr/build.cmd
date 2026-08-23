@echo off
rem Build the xess_xpu_corr SYCL extension in-place.
rem
rem Environment resolution order (no machine-specific paths are hardcoded):
rem   PYTHON_EXE   python used to build/load the extension (default: python)
rem   VSINSTALLDIR install root of Visual Studio / VS Build Tools
rem   ONEAPI_ROOT  Intel oneAPI root (default: %ProgramFiles(x86)%\Intel\oneAPI)
rem Usage: build.cmd [extra setup.py args]
setlocal

if not defined PYTHON_EXE set "PYTHON_EXE=python"

rem ---- MSVC host environment ------------------------------------------------
set "DISTUTILS_USE_SDK=1"
if defined VSINSTALLDIR if exist "%VSINSTALLDIR%\Common7\Tools\VsDevCmd.bat" (
    call "%VSINSTALLDIR%\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
    if errorlevel 1 exit /b 1
    goto oneapi
)
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VSDIR=%%i"
if defined VSDIR if exist "%VSDIR%\Common7\Tools\VsDevCmd.bat" (
    call "%VSDIR%\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
    if errorlevel 1 exit /b 1
    goto oneapi
)
echo [xpu_corr] VsDevCmd.bat not found. Install VS Build Tools with the C++ toolset, or set VSINSTALLDIR. 1>&2
exit /b 1

:oneapi
rem ---- DPC++ compiler environment -------------------------------------------
if not defined ONEAPI_ROOT set "ONEAPI_ROOT=%ProgramFiles(x86)%\Intel\oneAPI"
if exist "%ONEAPI_ROOT%\setvars.bat" call "%ONEAPI_ROOT%\setvars.bat" >nul 2>&1
where icx >nul 2>nul
if errorlevel 1 (
    echo [xpu_corr] icx.exe not found after sourcing oneAPI setvars.bat. Install the Intel oneAPI DPC++ Compiler or point ONEAPI_ROOT at it. 1>&2
    exit /b 1
)

rem ---- Ninja -----------------------------------------------------------------
where ninja >nul 2>nul
if errorlevel 1 if defined VSINSTALLDIR set "PATH=%PATH%;%VSINSTALLDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
where ninja >nul 2>nul
if errorlevel 1 (
    echo [xpu_corr] ninja.exe not found on PATH. 1>&2
    exit /b 1
)

cd /d "%~dp0"
"%PYTHON_EXE%" setup.py build_ext --inplace %*
exit /b %errorlevel%
