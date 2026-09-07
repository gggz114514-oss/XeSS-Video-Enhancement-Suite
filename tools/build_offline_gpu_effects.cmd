@echo off
setlocal enabledelayedexpansion
if "%~1"=="" exit /b 2
if not defined XESS_TOOLS_ROOT exit /b 2
if not defined VSCMD_PATH exit /b 2
if not defined OPENCL_CPP_ROOT exit /b 2
if not defined OPENCL_INCLUDE exit /b 2
if not defined OPENCL_LIBRARY exit /b 2
for %%I in ("%~dp0..") do set "REPO=%%~fI"
set "OUT=%~f1"
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%OUT%\temp" mkdir "%OUT%\temp"
set "TEMP=%OUT%\temp"
set "TMP=%TEMP%"
call "%VSCMD_PATH%" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
if not defined DXC_PATH set "DXC_PATH=%WindowsSdkDir%bin\%WindowsSDKVersion%x64\dxc.exe"
if not exist "%DXC_PATH%" exit /b 2
set "OV=%XESS_TOOLS_ROOT%\venv\Lib\site-packages\openvino"
set "SDK=%XESS_TOOLS_ROOT%\sdk\official"
if defined XESS_SDK_ROOT set "SDK=%XESS_SDK_ROOT%"
set "VPL=%XESS_TOOLS_ROOT%\tmp\libvpl-src"
set "VPL_LIB=%XESS_TOOLS_ROOT%\tmp\libvpl-build\Release"
pushd "%OUT%"
"%DXC_PATH%" -nologo -T cs_6_0 -E main "%REPO%\src\realtime\shaders\native_sr_effects.hlsl" -Fo native_sr_effects.dxil >build-effects-shader.log 2>&1
if errorlevel 1 (type build-effects-shader.log & popd & exit /b 1)
cl /nologo /O2 /std:c++17 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS /c /Fo:native_strict_dis_provider.obj "%REPO%\src\realtime\native_strict_dis_provider.cpp" >build-dis-provider.log 2>&1
if errorlevel 1 (type build-dis-provider.log & popd & exit /b 1)
for %%M in (vpl_gpu_full_fg providers\strict_dis_main) do (
 set "EXE=gpu-block-native.exe"
 set "FP="
 set "OBJ="
 if "%%M"=="providers\strict_dis_main" (
  set "EXE=gpu-dis-native.exe"
  set "FP=/fp:precise"
  set "OBJ=native_strict_dis_provider.obj"
 )
 cl /nologo /O2 !FP! /std:c++17 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS ^
 /I"%REPO%\src\realtime" /I"%VPL%\api\vpl" /I"%VPL%\api" /I"%OV%\include" ^
 /I"%OPENCL_CPP_ROOT%" /I"%OPENCL_INCLUDE%" /I"%SDK%\inc" ^
 /Fe:!EXE! "%REPO%\src\realtime\%%M.cpp" !OBJ! ^
 /link d3d11.lib d3d12.lib d3dcompiler.lib dxgi.lib ole32.lib user32.lib windowsapp.lib ^
 "%VPL_LIB%\vpl.lib" "%SDK%\lib\libxess.lib" "%SDK%\lib\libxess_fg.lib" "%SDK%\lib\libxell.lib" ^
 "%OV%\libs\openvino.lib" "%OPENCL_LIBRARY%" /SUBSYSTEM:CONSOLE >build-!EXE!.log 2>&1
 if errorlevel 1 (type build-!EXE!.log & popd & exit /b 1)
)
for %%F in (gpu-block-native.exe gpu-dis-native.exe native_sr_effects.dxil) do certutil -hashfile "%%F" SHA256
popd
exit /b 0
