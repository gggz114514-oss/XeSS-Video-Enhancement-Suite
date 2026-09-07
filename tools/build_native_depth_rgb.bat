@echo off
setlocal
if "%~1"=="" (echo Usage: build_native_depth_rgb.bat SHADER_OUTPUT_DIR & exit /b 1)
if not defined FXC set "FXC=%WindowsSdkDir%bin\%WindowsSDKVersion%x64\fxc.exe"
if not exist "%FXC%" (echo FXC unavailable; initialize VS/Windows SDK or set FXC & exit /b 1)
for %%I in ("%~dp0..") do set "REPO=%%~fI"
if not exist "%~1" mkdir "%~1"
for %%S in (native_depth_rgb native_depth_rgb_tensor) do (
 "%FXC%" /nologo /T cs_5_0 /E main /O3 /Ges /Gis /Fo "%~1\%%S.cso" "%REPO%\src\realtime\shaders\%%S.hlsl"
 if errorlevel 1 exit /b 1
)
exit /b 0
