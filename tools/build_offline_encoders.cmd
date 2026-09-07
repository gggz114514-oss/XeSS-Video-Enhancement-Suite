@echo off
setlocal enabledelayedexpansion
if "%~1"=="" (echo Usage: build_offline_encoders OUTPUT_DIRECTORY & exit /b 2)
if not defined XESS_TOOLS_ROOT (echo XESS_TOOLS_ROOT is required & exit /b 2)
if not defined AMD_SDK (echo AMD_SDK is required & exit /b 2)
for %%I in ("%~dp0..") do set "REPO=%%~fI"
set "OUT=%~f1"
call "%REPO%\tools\build_amd_of_native.cmd" "%AMD_SDK%" "%OUT%"
if errorlevel 1 exit /b 1
call "%VSCMD_PATH%" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
set "OV=%XESS_TOOLS_ROOT%\venv\Lib\site-packages\openvino"
set "SDK=%XESS_TOOLS_ROOT%\sdk\official"
if defined XESS_SDK_ROOT set "SDK=%XESS_SDK_ROOT%"
set "VPL=%XESS_TOOLS_ROOT%\tmp\libvpl-src"
set "VPL_LIB=%XESS_TOOLS_ROOT%\tmp\libvpl-build\Release"
pushd "%OUT%"
cl /nologo /W4 /O2 /std:c++17 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS /c /Fo:native_strict_dis_provider.obj "%REPO%\src\realtime\native_strict_dis_provider.cpp" >build-dis-provider.log 2>&1
if errorlevel 1 (type build-dis-provider.log & popd & exit /b 1)
cl /nologo /W4 /O2 /fp:precise /std:c++17 /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS ^
 /I"%REPO%\src\realtime" /I"%VPL%\api\vpl" /I"%VPL%\api" /I"%OV%\include" ^
 /I"%OPENCL_CPP_ROOT%" /I"%OPENCL_INCLUDE%" /I"%SDK%\inc" ^
 /Fo:strict_dis_native.obj /Fe:gpu-dis-native.exe "%REPO%\src\realtime\providers\strict_dis_main.cpp" native_strict_dis_provider.obj ^
 /link d3d11.lib d3d12.lib d3dcompiler.lib dxgi.lib ole32.lib user32.lib windowsapp.lib ^
 "%VPL_LIB%\vpl.lib" "%SDK%\lib\libxess.lib" "%SDK%\lib\libxess_fg.lib" "%SDK%\lib\libxell.lib" ^
 "%OV%\libs\openvino.lib" "%OPENCL_LIBRARY%" /SUBSYSTEM:CONSOLE >build-dis-native.log 2>&1
if errorlevel 1 (type build-dis-native.log & popd & exit /b 1)
copy /Y gpu-block-control.exe gpu-block-native.exe >nul
for %%F in (gpu-block-native.exe gpu-dis-native.exe amd-of-native.exe) do certutil -hashfile "%%F" SHA256
popd
exit /b 0
