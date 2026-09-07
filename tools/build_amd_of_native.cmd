@echo off
setlocal enabledelayedexpansion
if "%~2"=="" (echo Usage: build_amd_of_native SDK_CHECKOUT OUT_DIRECTORY & exit /b 2)
set "AMD_SDK=%~f1"
set "OUT=%~f2"
for %%I in ("%~dp0..") do set "REPO=%%~fI"
for %%I in ("%REPO%\..\..\..") do set "BASE=%%~fI"
if defined XESS_TOOLS_ROOT set "BASE=%XESS_TOOLS_ROOT%"
if not defined VSCMD_PATH (echo VSCMD_PATH is required & exit /b 2)
if not defined DXC_PATH (echo DXC_PATH is required & exit /b 2)
if not defined AMD_GENERATED_ROOT (echo AMD_GENERATED_ROOT must name the pinned FP32 shader headers & exit /b 2)
if not defined OPENCL_CPP_ROOT (echo OPENCL_CPP_ROOT is required & exit /b 2)
if not defined OPENCL_INCLUDE (echo OPENCL_INCLUDE is required & exit /b 2)
if not defined OPENCL_LIBRARY (echo OPENCL_LIBRARY is required & exit /b 2)
set "XSDK=%BASE%\sdk\official"
if defined XESS_SDK_ROOT set "XSDK=%XESS_SDK_ROOT%"
set "VPL=%BASE%\tmp\libvpl-src"
set "VPL_LIB=%BASE%\tmp\libvpl-build\Release"
set "OV=%BASE%\venv\Lib\site-packages\openvino"
if defined OPENVINO_ROOT set "OV=%OPENVINO_ROOT%"
if not exist "%AMD_GENERATED_ROOT%\ffx_opticalflow_prepare_luma_pass_permutations.h" exit /b 2
if not exist "%AMD_SDK%\Kits\FidelityFX\framegeneration\fsr3\internal\ffx_opticalflow.cpp" exit /b 2
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%OUT%\shaders" mkdir "%OUT%\shaders"
call "%VSCMD_PATH%" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
call "%REPO%\tools\build_native_depth_rgb.bat" "%OUT%\shaders"
if errorlevel 1 exit /b 1
pushd "%OUT%"
for %%S in (surface_rgba_luma surface_nv12_color surface_gpu_motion_tile surface_gpu_motion_repair surface_gpu_velocity surface_gpu_mask surface_gpu_lite_downsample surface_gpu_lite_upsample native_depth_reduce native_depth_normalize native_depth_temporal native_depth_resize native_depth_summary native_scene_stats native_terminal_post native_sr_effects native_provider_adapter native_amd_dense) do (
 "%DXC_PATH%" -nologo -T cs_6_0 -E main "%REPO%\src\realtime\shaders\%%S.hlsl" -Fo "%OUT%\shaders\%%S.dxil" >>build-shaders.log 2>&1
 if errorlevel 1 (type build-shaders.log & popd & exit /b 1)
)
cl /nologo /c /O2 /std:c++20 /utf-8 /EHsc /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DNDEBUG ^
 /I"%AMD_SDK%\Kits\FidelityFX\api\internal" /FIffx_util.h ^
 /I"%AMD_SDK%\Kits\FidelityFX\backend\dx12" /I"%AMD_SDK%\Kits\FidelityFX\framegeneration\fsr3\include" /I"%AMD_GENERATED_ROOT%" ^
 "%REPO%\src\realtime\amd_of_sdk_backend.cpp" ^
 "%AMD_SDK%\Kits\FidelityFX\framegeneration\fsr3\internal\ffx_opticalflow.cpp" ^
 "%AMD_SDK%\Kits\FidelityFX\framegeneration\fsr3\internal\ffx_opticalflow_shaderblobs.cpp" ^
 "%AMD_SDK%\Kits\FidelityFX\api\internal\ffx_assert.cpp" "%AMD_SDK%\Kits\FidelityFX\api\internal\ffx_object_management.cpp" ^
 "%AMD_SDK%\Kits\FidelityFX\api\internal\ffx_message.cpp" "%AMD_SDK%\Kits\FidelityFX\backend\dx12\ffx_backends_dx12.cpp" >build-sdk.log 2>&1
if errorlevel 1 (type build-sdk.log & popd & exit /b 1)
set "COMMON=/nologo /O2 /std:c++17 /utf-8 /EHsc /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DNDEBUG"
for %%M in (amd_of_native_main vpl_gpu_full_fg) do (
 set "EXE=amd-of-native.exe"
 if "%%M"=="vpl_gpu_full_fg" set "EXE=gpu-block-control.exe"
 cl !COMMON! /I"%VPL%\api\vpl" /I"%VPL%\api" /I"%XSDK%\inc" ^
 /I"%OV%\include" /I"%OPENCL_CPP_ROOT%" /I"%OPENCL_INCLUDE%" ^
 /I"%AMD_SDK%\Kits\FidelityFX\api\internal" /FIffx_util.h ^
 /I"%AMD_SDK%\Kits\FidelityFX\backend\dx12" /I"%AMD_SDK%\Kits\FidelityFX\framegeneration\fsr3\include" /I"%AMD_GENERATED_ROOT%" ^
 "%REPO%\src\realtime\%%M.cpp" ^
 amd_of_sdk_backend.obj ffx_opticalflow.obj ffx_opticalflow_shaderblobs.obj ffx_assert.obj ffx_object_management.obj ffx_message.obj ffx_backends_dx12.obj ^
 /Fe:!EXE! /link d3d11.lib d3d12.lib dxgi.lib dxguid.lib d3dcompiler.lib ole32.lib user32.lib windowsapp.lib ^
 "%VPL_LIB%\vpl.lib" "%XSDK%\lib\libxess.lib" "%XSDK%\lib\libxess_fg.lib" "%XSDK%\lib\libxell.lib" ^
 "%OV%\libs\openvino.lib" "%OPENCL_LIBRARY%" >build-%%M.log 2>&1
 if errorlevel 1 (type build-%%M.log & popd & exit /b 1)
)
copy /Y "%VPL_LIB%\libvpl.dll" "%OUT%" >nul
for %%D in (libxess libxess_fg libxell) do copy /Y "%XSDK%\bin\%%D.dll" "%OUT%" >nul
for %%D in (openvino openvino_intel_gpu_plugin openvino_ir_frontend tbb12 tbbmalloc) do copy /Y "%OV%\libs\%%D.dll" "%OUT%" >nul
certutil -hashfile amd-of-native.exe SHA256
echo BUILD OK
popd
endlocal
