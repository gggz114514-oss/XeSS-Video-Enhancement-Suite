@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "SDKROOT=%XESS_SDK_ROOT%"
if not defined SDKROOT set "SDKROOT=%ROOT%.runtime\engine\sdk\official"
if not exist "%SDKROOT%\inc\xess\xess_d3d12.h" (
    echo [build] XeSS SDK headers not found.
    echo [build] Set XESS_SDK_ROOT to the SDK directory containing inc, lib and bin.
    exit /b 1
)

if not defined VSCMD_VER (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "%VSWHERE%" (
        for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%I"
        if defined VSROOT call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
    )
)
where cl.exe >nul 2>nul
if errorlevel 1 (
    echo [build] Visual Studio 2022 C++ Build Tools not found.
    exit /b 1
)

if not exist build mkdir build
cl /nologo /W3 /O2 /std:c++17 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS src\xess_vsr.cpp /I "%SDKROOT%\inc" /Fo:build\xess_vsr.obj /Fe:build\xess-vsr.exe /link d3d12.lib dxgi.lib "%SDKROOT%\lib\libxess.lib" /SUBSYSTEM:CONSOLE
if errorlevel 1 goto :error
cl /nologo /W4 /O2 /std:c++17 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS src\xess_fg.cpp /I "%SDKROOT%\inc" /Fo:build\xess_fg.obj /Fe:build\xess-fg.exe /link d3d12.lib d3d11.lib dxgi.lib user32.lib windowsapp.lib "%SDKROOT%\lib\libxess_fg.lib" "%SDKROOT%\lib\libxell.lib" /SUBSYSTEM:CONSOLE
if errorlevel 1 goto :error

set "DLLROOT=%SDKROOT%\bin"
if not exist "%DLLROOT%\libxess.dll" set "DLLROOT=%SDKROOT%\..\.."
if exist "%DLLROOT%\libxess.dll" copy /Y "%DLLROOT%\libxess.dll" build\ >nul
if exist "%DLLROOT%\libxess_fg.dll" copy /Y "%DLLROOT%\libxess_fg.dll" build\ >nul
if exist "%DLLROOT%\libxell.dll" copy /Y "%DLLROOT%\libxell.dll" build\ >nul
echo [build] OK
exit /b 0

:error
echo [build] FAILED
exit /b 1
