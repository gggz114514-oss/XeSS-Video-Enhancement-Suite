# Third-party components added for AI depth

## Depth Anything V2 Small

- Source model: `depth-anything/Depth-Anything-V2-Small-hf`
- Project: https://github.com/DepthAnything/Depth-Anything-V2
- Model page: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- License declared by the model page: Apache License 2.0

The packaged `depth_anything_v2_small.xml/.bin` files are an FP16 OpenVINO IR
conversion of that checkpoint. `model.json` records the source model identifier
and preprocessing constants.

## OpenVINO Runtime

- Project: https://github.com/openvinotoolkit/openvino
- License: Apache License 2.0

The Python wheel's license and metadata files are retained under
`python/Lib/site-packages/openvino-*.dist-info`.

## Intel XeSS / XeSS-FG / XeLL SDK 2.1

- Vendor: Intel Corporation
- SDK components: `libxess.dll`, `libxess_fg.dll`, `libxell.dll`
- Project: https://github.com/intel/xess

The binaries are redistributed under the license supplied with Intel XeSS SDK
2.1.0. Applications use only the documented XeSS, XeFG Swap Chain, and XeLL
interfaces; no reverse-engineered core FG structures are used.
