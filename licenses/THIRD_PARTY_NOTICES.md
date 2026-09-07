# R4 runtime third-party inventory

Components remain under their respective upstream licenses; these notices do not relicense the project's own code.

- **Intel XeSS / XeFG / XeLL**: SDK supplied by https://github.com/intel/xess . The exact supplied LICENSE and third-party-programs texts accompany the DLLs. Public swap-chain APIs are used; no reverse-engineered private FG structures.
- **Intel oneVPL**: https://github.com/intel/libvpl , MIT license and upstream third-party notices included as ONEVPL_LICENSE.txt / ONEVPL_THIRD_PARTY.txt.
- **OpenVINO 2025.4.1**: https://github.com/openvinotoolkit/openvino . The wheel's dist-info/licenses are retained. No torch package is added to ComfyUI.
- **Depth Anything V2 Small**: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf and https://github.com/DepthAnything/Depth-Anything-V2 . FP16 OpenVINO IR conversion, not a new model; model.json records preprocessing. Small model is the packaged family; not the other differently licensed sizes.
- **OpenCV 5.0 / DIS reference**: https://github.com/opencv/opencv , including `modules/video/src/dis_flow.cpp`, `variational_refinement.cpp`, and `opencl/dis_flow.cl`. The GPU DIS port follows/changes these algorithms for D3D12; it is not claimed to be an independent original algorithm. Apache-2.0 text is included; Python wheel notices are retained. GPU Block is a separate project-developed estimator, not identical to OpenCV DIS.
- **AMD FidelityFX Optical Flow**: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK . MIT license text from the included SDK sources is in AMD_FIDELITYFX_MIT.txt. AMD here names the algorithm, not whole-pipeline hardware certification.
- **Python 3.13.11 / NumPy 2.3.5 / packaging 26.3**: https://github.com/python/cpython/tree/v3.13.11 , https://github.com/numpy/numpy/tree/v2.3.5 , https://github.com/pypa/packaging . Python license and wheel metadata/licenses are retained in the runtime.
- **Microsoft VC++ runtime**: app-local redistributable files from the MSVC 2022 Redist directory; supplied REDIST list included. Not a replacement for the user's GPU driver or Windows components.

## FFmpeg tools and corresponding source locations

These are separate executables/libraries, not relabeled as proprietary XeSS code. The build configurations can be inspected with `-version`.

- `media/ffmpeg.exe`: unmodified **7.1-essentials_build-www.gyan.dev**, GPLv3 configuration, including x264/x265. Binary distributor and build/source links: https://www.gyan.dev/ffmpeg/builds/ and https://github.com/GyanD/codexffmpeg/releases . FFmpeg source for this release: https://github.com/FFmpeg/FFmpeg/tree/n7.1 (archive https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n7.1.tar.gz ). GPLv3 text is included as FFMPEG_GPLv3.txt. Preserve upstream copyright and corresponding source/build information when redistributing.
- `probe/ffprobe.exe` + shared DLLs: unmodified **n9.0.1-11-ge47273f4d9-20260829**, LGPL shared configuration from https://github.com/BtbN/FFmpeg-Builds . FFmpeg source: https://github.com/FFmpeg/FFmpeg/commit/e47273f4d9 ; build recipes and dependency sources: https://github.com/BtbN/FFmpeg-Builds . Supplied LGPL license is FFPROBE_LGPL_LICENSE.txt. No GPL/nonfree options are added to this probe build.

Runtime packages carry these licenses and the Python wheel notices. Upstream source versions are pinned for reproducibility; changing FFmpeg versions requires rechecking the color-conversion calibration and media tests, not just replacing an EXE.
