@echo off
setlocal
if "%~1"=="" exit /b 2
if not defined DXC_PATH (echo DXC_PATH is required & exit /b 2)
for %%I in ("%~dp0..") do set "REPO=%%~fI"
set "OUT=%~f1"
if not exist "%OUT%" mkdir "%OUT%"
"%DXC_PATH%" -nologo -T cs_6_0 -E main "%REPO%\src\realtime\shaders\native_sr_effects.hlsl" -Fo "%OUT%\native_sr_effects.dxil"
if errorlevel 1 exit /b 1
echo [build] Public SR effects shader built. Native GPU workers are binary-only.
exit /b 0
