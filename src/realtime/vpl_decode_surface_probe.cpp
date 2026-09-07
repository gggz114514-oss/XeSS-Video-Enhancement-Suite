// Gate 1 probe: decode compressed H.264/H.265 into oneVPL D3D11 surfaces and
// prove the native texture can be opened by a same-LUID D3D12 device.  This is
// intentionally a probe, not a product decoder: compressed input may be read
// into system memory, but no decoded pixel surface is mapped or written out.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define ONEVPL_EXPERIMENTAL

#include <windows.h>
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <vpl/mfxdispatcher.h>
#include <vpl/mfxmemory.h>
#include <vpl/mfxvideo.h>

using Microsoft::WRL::ComPtr;

namespace {

constexpr mfxU32 kApiVersion = (2u << 16) | 9u;
constexpr std::size_t kMaxCompressedInput = 256u * 1024u * 1024u;

struct Options {
    std::string input;
    std::string codec = "hevc";
    std::string report;
    std::string shader_dir = "build";
    // Optional derived-probe diagnostic output.  This is intentionally not a
    // Gate 1/product path: callers that request it explicitly accept a
    // full-frame readback for quality inspection.
    std::string diagnostic_output_raw;
    // Optional flow-only diagnostics.  These are never enabled by the
    // product path; requested fields are converted to float32 on readback so
    // CPU warp tools do not mistake the native R16G16_FLOAT layout for RGB.
    std::string diagnostic_backward_flow_raw;
    std::string diagnostic_forward_flow_raw;
    // Optional input-resolution responsive-mask diagnostic.  Like flow
    // diagnostics this is an explicit readback and is never enabled by the
    // GPU-resident product path.
    std::string diagnostic_mask_raw;
    // Optional input-resolution RGBA color-pass dump.  Quality evidence for
    // the NV12->RGB shader only; an explicit readback the product path never
    // enables.
    std::string diagnostic_color_raw;
    // Color conversion follows compressed-stream signal metadata in auto
    // mode. Explicit values are retained for controlled A/B validation.
    std::string color_matrix = "auto";
    std::string color_range = "auto";
    // Motion repair mode for the H2 tiled search: off keeps the raw field
    // (ablation baseline), propagate is repair A (trusted-median or zero),
    // refine is repair B (outlier-only joint re-search).
    std::string motion_repair = "off";
    // Full-GPU FG motion provider.  gpu-block remains the default product
    // path; gpu-dis selects the strict GPU DIS graph imported by the FG
    // executable.  Other probes ignore this field.
    std::string motion_backend = "gpu-block";
    // Full-GPU chain (full-chain probe only): QSV output path, fused GPU
    // post switch, pipeline slot count and the input frame rate for PTS.
    std::string qsv_out;
    std::string terminal_encoder = "h264_qsv";
    // Optional precomputed OpenVINO inverse-depth planes. Full-GPU FG reads
    // these planes only to upload them into per-slot D3D12 depth textures.
    std::string depth_dir;
    std::string depth_model;
    std::string gpu_depth_input = "ffmpeg-rgb";
    std::string diagnostic_core_dir;
    double stability_seconds=0;
    bool loop_input=false;
    bool synthetic_external=false;
    int synthetic_gap_at=-1;
    std::string cancel_file;
    std::string gpu_mode = "sr-fg";
    std::string gpu_post = "on";
    std::string gpu_five_frame = "off";
    std::string gpu_anti_stripe = "off";
    int slot_count = 4;
    double fps = 0.0;
    // Optional Tikhonov-style center bias (normalized-luma SAD per pixel of
    // displacement) that breaks flat-region search ties coherently.
    float motion_center_bias = 0.0f;
    // GPU Block Lite motion grid scale.  Full=1.0 keeps the existing search
    // grid; values below one run the same search kernel on a reduced luma
    // grid and restore flow vectors to the original input grid on GPU.
    float motion_scale = 1.0f;
    int motion_aux_radius = 0;
    // Optional consumer parameters used by the Gate 3/4 surface probes.
    // Gate 1 itself ignores these fields; keeping them in the shared parser
    // lets the derived probe use exactly the same compressed-surface reader.
    std::string xess_quality = "performance";
    int output_width = 0;
    int output_height = 0;
    int adapter = 0;
    int max_frames = 8;
    // Optional realtime display endpoint.  The default keeps all existing
    // probes unchanged; realtime-video-probe enables one of these values to
    // Present a GPU resource directly to a D3D12 swap chain.
    std::string display_mode = "none";
    bool display_visible = true;
};

// One record per decoder surface lease.  The old probe only retained aggregate
// success counts, which was insufficient to audit the native-handle contract:
// a successful frame must show the exact QI/CreateSharedHandle/OpenSharedHandle
// HRESULTs and the release ordering that returns the lease to oneVPL.
struct SurfaceLeaseRecord {
    int frame = -1;
    std::string synchronize_status;
    std::string export_status;
    std::string qi_hr;
    std::string create_shared_handle_hr;
    std::string open_shared_handle_hr;
    std::string source_desc;
    std::string opened_desc;
    bool decoder_lease_acquired = false;
    bool exported_lease_acquired = false;
    bool exported_lease_released = false;
    bool decoder_lease_released = false;
};

struct Result {
    std::string input;
    std::string codec;
    std::string adapter_name;
    std::string adapter_luid;
    std::string d3d11_luid;
    std::string d3d12_luid;
    std::string fourcc;
    mfxU16 width = 0;
    mfxU16 height = 0;
    mfxU16 crop_w = 0;
    mfxU16 crop_h = 0;
    int requested = 0;
    int decoded = 0;
    int exported = 0;
    int native_texture_seen = 0;
    int shared_handle_created = 0;
    int d3d12_opened = 0;
    int d3d12_copy_count = 0;
    std::uint64_t cpu_readback_bytes = 0;
    std::uint64_t cpu_pixel_upload_bytes = 0;
    std::uint64_t diagnostic_output_readback_bytes = 0;
    std::uint64_t diagnostic_backward_flow_readback_bytes = 0;
    std::uint64_t diagnostic_forward_flow_readback_bytes = 0;
    std::uint64_t diagnostic_mask_readback_bytes = 0;
    std::uint64_t diagnostic_color_readback_bytes = 0;
    double first_frame_backward_mv_max_abs = -1.0;
    double first_frame_forward_mv_max_abs = -1.0;
    double first_frame_mask_min = -1.0;
    double first_frame_mask_max = -1.0;
    // Optional D3D12 product-observation fields populated by derived probes.
    // Zero means the adapter did not expose QueryVideoMemoryInfo.
    std::uint64_t vram_budget_bytes = 0;
    std::uint64_t vram_usage_before_bytes = 0;
    std::uint64_t vram_usage_after_bytes = 0;
    std::uint64_t vram_usage_peak_bytes = 0;
    double first_frame_wall_ms = -1.0;
    double total_wall_ms = -1.0;
    std::string display_mode = "none";
    int display_window_created = 0;
    int display_presented_frames = 0;
    int display_dropped_frames = 0;
    int display_copy_count_total = 0;
    std::uint64_t display_cpu_readback_bytes = 0;
    std::uint64_t display_cpu_upload_bytes = 0;
    double display_first_frame_wall_ms = -1.0;
    double display_total_wall_ms = -1.0;
    double display_present_fps = -1.0;
    double display_present_copy_barrier_ms = -1.0;
    double display_input_fps = -1.0;
    int display_pts_available = 0;
    int display_pts_discontinuities = 0;
    // V3 wall-clock attribution. These are accumulated over the run and
    // intentionally describe CPU spans, not a sum that should be treated as
    // end-to-end latency because several spans can overlap GPU execution.
    int cpu_timed_frames = 0;
    int cpu_decode_calls = 0;
    int cpu_fence_wait_count = 0;
    int cpu_present_call_count = 0;
    double cpu_frame_span_ms = 0.0;
    double cpu_bitstream_feed_ms = 0.0;
    double cpu_decode_frame_async_ms = 0.0;
    double cpu_surface_sync_ms = 0.0;
    double cpu_native_shared_open_ms = 0.0;
    double cpu_resource_setup_ms = 0.0;
    double cpu_allocator_reset_ms = 0.0;
    double cpu_command_record_ms = 0.0;
    double cpu_execute_command_lists_ms = 0.0;
    double cpu_fence_wait_ms = 0.0;
    double cpu_display_allocator_reset_ms = 0.0;
    double cpu_display_command_record_ms = 0.0;
    double cpu_present_call_ms = 0.0;
    double cpu_present_fence_wait_ms = 0.0;
    double cpu_surface_release_ms = 0.0;
    double cpu_frame_tail_reclaim_ms = 0.0;
    std::string color_matrix_requested = "auto";
    std::string color_range_requested = "auto";
    std::string color_matrix_selected;
    std::string color_range_selected;
    std::string color_matrix_fallback_reason;
    std::string color_range_fallback_reason;
    std::string motion_repair_selected;
    float motion_scale = 1.0f;
    int motion_aux_radius = 0;
    int signal_info_available = 0;
    int signal_video_format = -1;
    int signal_video_full_range = -1;
    int signal_colour_description_present = -1;
    int signal_colour_primaries = -1;
    int signal_transfer_characteristics = -1;
    int signal_matrix_coefficients = -1;
    std::vector<SurfaceLeaseRecord> surface_leases;
    std::string block;
};

enum class ColorMatrix { Bt601, Bt709 };
enum class ColorRange { Limited, Full };

const char *ColorMatrixName(ColorMatrix value) {
    return value == ColorMatrix::Bt601 ? "bt601" : "bt709";
}

const char *ColorRangeName(ColorRange value) {
    return value == ColorRange::Full ? "full" : "limited";
}

bool ParseColorMatrix(const std::string &value) {
    return value == "auto" || value == "bt601" || value == "bt709";
}

bool ParseColorRange(const std::string &value) {
    return value == "auto" || value == "limited" || value == "full";
}

bool ParseMotionRepair(const std::string &value) {
    return value == "off" || value == "propagate" || value == "refine";
}

struct ColorSelection {
    ColorMatrix matrix = ColorMatrix::Bt709;
    ColorRange range = ColorRange::Limited;
    bool matrix_from_signal = false;
    bool range_from_signal = false;
    // Why a selection was made when the bitstream did not decide it.
    // Empty string means no fallback happened (VUI metadata or CLI decided).
    std::string matrix_fallback_reason;
    std::string range_fallback_reason;
};

ColorSelection SelectColor(const Options &options, const mfxExtVideoSignalInfo &signal,
                           int width, int height) {
    ColorSelection selection;
    (void)width;
    (void)height;
    if (options.color_matrix == "bt601") {
        selection.matrix = ColorMatrix::Bt601;
    } else if (options.color_matrix == "bt709") {
        selection.matrix = ColorMatrix::Bt709;
    } else if (signal.ColourDescriptionPresent && signal.MatrixCoefficients == 1) {
        // ITU-R BT.709 matrix explicitly signalled in the H.264 VUI.
        selection.matrix = ColorMatrix::Bt709;
        selection.matrix_from_signal = true;
    } else if (signal.ColourDescriptionPresent && signal.MatrixCoefficients == 6) {
        // SMPTE 170M is the H.264 code used for BT.601/NTSC.
        selection.matrix = ColorMatrix::Bt601;
        selection.matrix_from_signal = true;
    } else if (signal.ColourDescriptionPresent && signal.MatrixCoefficients == 5) {
        // BT.470BG is the PAL/BT.601-family matrix.
        selection.matrix = ColorMatrix::Bt601;
        selection.matrix_from_signal = true;
    } else {
        // No usable matrix metadata.  Calibrated against FFmpeg 7.1 on this
        // machine: swscale converts with ff_yuv2rgb_coeffs[SWS_CS_DEFAULT]
        // (BT.601) at every resolution when the H.264 VUI carries no colour
        // description.  Verified with synthetic x264 streams from 256x256 up
        // to 1920x1080: the decoded RGB24 is byte-equal to the BT.601
        // limited-range model at all sizes, so geometry must NOT pick
        // BT.709 for HD; the real 1080p C material has no VUI either.
        selection.matrix = ColorMatrix::Bt601;
        if (signal.ColourDescriptionPresent)
            selection.matrix_fallback_reason =
                "unsupported_vui_matrix_coefficients_" +
                std::to_string(signal.MatrixCoefficients) + "_default_601";
        else
            selection.matrix_fallback_reason =
                "vui_colour_description_absent_default_601";
    }
    if (options.color_range == "full") {
        selection.range = ColorRange::Full;
    } else if (options.color_range == "limited") {
        selection.range = ColorRange::Limited;
    } else if (signal.VideoFullRange == 1) {
        selection.range = ColorRange::Full;
        selection.range_from_signal = true;
    } else {
        selection.range_fallback_reason = "vui_full_range_flag_absent_limited";
    }
    return selection;
}

struct ComRuntime {
    HRESULT hr = E_FAIL;
    ComRuntime() : hr(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) {}
    ~ComRuntime() {
        if (SUCCEEDED(hr)) CoUninitialize();
    }
    bool usable() const { return SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE; }
};

const char *StatusName(mfxStatus status) {
    switch (status) {
        case MFX_ERR_NONE: return "MFX_ERR_NONE";
        case MFX_ERR_UNKNOWN: return "MFX_ERR_UNKNOWN";
        case MFX_ERR_NULL_PTR: return "MFX_ERR_NULL_PTR";
        case MFX_ERR_UNSUPPORTED: return "MFX_ERR_UNSUPPORTED";
        case MFX_ERR_MEMORY_ALLOC: return "MFX_ERR_MEMORY_ALLOC";
        case MFX_ERR_NOT_ENOUGH_BUFFER: return "MFX_ERR_NOT_ENOUGH_BUFFER";
        case MFX_ERR_INVALID_HANDLE: return "MFX_ERR_INVALID_HANDLE";
        case MFX_ERR_LOCK_MEMORY: return "MFX_ERR_LOCK_MEMORY";
        case MFX_ERR_NOT_INITIALIZED: return "MFX_ERR_NOT_INITIALIZED";
        case MFX_ERR_MORE_DATA: return "MFX_ERR_MORE_DATA";
        case MFX_ERR_MORE_SURFACE: return "MFX_ERR_MORE_SURFACE";
        case MFX_ERR_ABORTED: return "MFX_ERR_ABORTED";
        case MFX_ERR_DEVICE_LOST: return "MFX_ERR_DEVICE_LOST";
        case MFX_ERR_DEVICE_FAILED: return "MFX_ERR_DEVICE_FAILED";
        case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM: return "MFX_ERR_INCOMPATIBLE_VIDEO_PARAM";
        case MFX_ERR_INVALID_VIDEO_PARAM: return "MFX_ERR_INVALID_VIDEO_PARAM";
        case MFX_WRN_IN_EXECUTION: return "MFX_WRN_IN_EXECUTION";
        case MFX_WRN_DEVICE_BUSY: return "MFX_WRN_DEVICE_BUSY";
        case MFX_WRN_VIDEO_PARAM_CHANGED: return "MFX_WRN_VIDEO_PARAM_CHANGED";
        default: return "MFX_STATUS_OTHER";
    }
}

std::string HexHr(HRESULT hr) {
    char out[32]{};
    std::snprintf(out, sizeof(out), "0x%08lX", static_cast<unsigned long>(hr));
    return out;
}

std::string LuidString(const LUID &luid) {
    char out[40]{};
    std::snprintf(out, sizeof(out), "%08lX:%08lX",
                  static_cast<unsigned long>(luid.HighPart),
                  static_cast<unsigned long>(luid.LowPart));
    return out;
}

const char *FourCCName(mfxU32 fourcc) {
    if (fourcc == MFX_FOURCC_NV12) return "NV12";
    if (fourcc == MFX_FOURCC_P010) return "P010";
    if (fourcc == MFX_FOURCC_P016) return "P016";
    return "other";
}

void Usage() {
    std::printf("Usage: vpl-decode-surface-probe --input <H264/H265 elementary stream> "
                "[--codec h264|hevc] [--max-frames N] [--adapter N] [--report PATH] "
                "[--diagnostic-output-raw PATH]\n");
    std::printf("The input is compressed-only CPU I/O. Decoded pixels never enter a CPU map/readback.\n");
}

bool ParseArgs(int argc, char **argv, Options &options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto take = [&](std::string &value) {
            if (i + 1 >= argc) return false;
            value = argv[++i];
            return !value.empty();
        };
        if (arg == "--input" || arg == "-i") {
            if (!take(options.input)) return false;
        } else if (arg == "--codec") {
            if (!take(options.codec)) return false;
            std::transform(options.codec.begin(), options.codec.end(), options.codec.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (options.codec != "h264" && options.codec != "avc" &&
                options.codec != "hevc" && options.codec != "h265")
                return false;
            if (options.codec == "avc") options.codec = "h264";
            if (options.codec == "h265") options.codec = "hevc";
        } else if (arg == "--max-frames") {
            std::string value;
            if (!take(value)) return false;
            options.max_frames = std::max(1, std::atoi(value.c_str()));
        } else if (arg == "--adapter") {
            std::string value;
            if (!take(value)) return false;
            options.adapter = std::max(0, std::atoi(value.c_str()));
        } else if (arg == "--report") {
            if (!take(options.report)) return false;
        } else if (arg == "--shader-dir") {
            if (!take(options.shader_dir)) return false;
        } else if (arg == "--diagnostic-output-raw") {
            if (!take(options.diagnostic_output_raw)) return false;
        } else if (arg == "--diagnostic-backward-flow-raw") {
            if (!take(options.diagnostic_backward_flow_raw)) return false;
        } else if (arg == "--diagnostic-forward-flow-raw") {
            if (!take(options.diagnostic_forward_flow_raw)) return false;
        } else if (arg == "--diagnostic-mask-raw") {
            if (!take(options.diagnostic_mask_raw)) return false;
        } else if (arg == "--diagnostic-color-raw") {
            if (!take(options.diagnostic_color_raw)) return false;
        } else if (arg == "--color-matrix") {
            if (!take(options.color_matrix)) return false;
            std::transform(options.color_matrix.begin(), options.color_matrix.end(),
                           options.color_matrix.begin(), [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            if (!ParseColorMatrix(options.color_matrix)) return false;
        } else if (arg == "--color-range") {
            if (!take(options.color_range)) return false;
            std::transform(options.color_range.begin(), options.color_range.end(),
                           options.color_range.begin(), [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            if (!ParseColorRange(options.color_range)) return false;
        } else if (arg == "--motion-repair") {
            if (!take(options.motion_repair)) return false;
            std::transform(options.motion_repair.begin(), options.motion_repair.end(),
                           options.motion_repair.begin(), [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            if (!ParseMotionRepair(options.motion_repair)) return false;
        } else if (arg == "--motion-backend") {
            if (!take(options.motion_backend)) return false;
            std::transform(options.motion_backend.begin(), options.motion_backend.end(),
                           options.motion_backend.begin(), [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            if (options.motion_backend != "gpu-block" &&
                options.motion_backend != "gpu-dis" &&
                options.motion_backend != "amd-of" &&
                options.motion_backend != "zero-mv") return false;
        } else if (arg == "--motion-center-bias") {
            std::string value;
            if (!take(value)) return false;
            options.motion_center_bias = static_cast<float>(atof(value.c_str()));
        } else if (arg == "--motion-scale") {
            std::string value;
            if (!take(value)) return false;
            options.motion_scale = static_cast<float>(atof(value.c_str()));
            if (!(options.motion_scale > 0.0f && options.motion_scale <= 1.0f))
                return false;
        } else if (arg == "--motion-aux-radius") {
            std::string value;
            if (!take(value)) return false;
            options.motion_aux_radius = atoi(value.c_str());
            if (options.motion_aux_radius < 1 || options.motion_aux_radius > 8)
                return false;
        } else if (arg == "--qsv-out") {
            if (!take(options.qsv_out)) return false;
        } else if (arg == "--terminal-encoder") {
            if (!take(options.terminal_encoder)) return false;
            if(options.terminal_encoder!="h264_qsv" && options.terminal_encoder!="hevc_qsv" &&
               options.terminal_encoder!="libx264" && options.terminal_encoder!="libx265" &&
               options.terminal_encoder!="ffv1") return false;
        } else if (arg == "--depth-dir") {
            if (!take(options.depth_dir)) return false;
        } else if (arg == "--depth-model") {
            if (!take(options.depth_model)) return false;
        } else if (arg == "--diagnostic-core-dir") {
            if (!take(options.diagnostic_core_dir)) return false;
        } else if (arg == "--stability-seconds") {
            std::string value;if(!take(value))return false;
            options.stability_seconds=std::atof(value.c_str());
            if(options.stability_seconds<=0 || options.stability_seconds>900)return false;
            options.loop_input=true;
        } else if (arg == "--synthetic-external") {
            options.synthetic_external=true;
        } else if (arg == "--synthetic-gap-at") {
            std::string value;if(!take(value))return false;
            options.synthetic_gap_at=std::atoi(value.c_str());
            if(options.synthetic_gap_at<1)return false;
        } else if (arg == "--cancel-file") {
            if(!take(options.cancel_file))return false;
        } else if (arg == "--gpu-mode") {
            if (!take(options.gpu_mode)) return false;
            if (options.gpu_mode != "sr" && options.gpu_mode != "fg" && options.gpu_mode != "sr-fg") return false;
        } else if (arg == "--gpu-depth-input") {
            if(!take(options.gpu_depth_input))return false;
            if(options.gpu_depth_input!="ffmpeg-rgb"&&options.gpu_depth_input!="legacy-nv12")return false;
        } else if (arg == "--gpu-five-frame" || arg == "--gpu-anti-stripe") {
            auto& value = arg == "--gpu-five-frame" ? options.gpu_five_frame : options.gpu_anti_stripe;
            if(!take(value) || (value!="on" && value!="off")) return false;
        } else if (arg == "--gpu-post") {
            if (!take(options.gpu_post)) return false;
            std::transform(options.gpu_post.begin(), options.gpu_post.end(),
                           options.gpu_post.begin(), [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            if (options.gpu_post != "on" && options.gpu_post != "off") return false;
        } else if (arg == "--slots") {
            std::string value;
            if (!take(value)) return false;
            options.slot_count = std::atoi(value.c_str());
        } else if (arg == "--fps") {
            std::string value;
            if (!take(value)) return false;
            options.fps = atof(value.c_str());
        } else if (arg == "--xess-quality") {
            if (!take(options.xess_quality)) return false;
            std::transform(options.xess_quality.begin(), options.xess_quality.end(),
                           options.xess_quality.begin(), [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            if (options.xess_quality != "performance" &&
                options.xess_quality != "balanced" &&
                options.xess_quality != "quality" &&
                options.xess_quality != "ultra-quality") return false;
        } else if (arg == "--output-width") {
            std::string value;
            if (!take(value)) return false;
            options.output_width = std::max(0, std::atoi(value.c_str()));
        } else if (arg == "--output-height") {
            std::string value;
            if (!take(value)) return false;
            options.output_height = std::max(0, std::atoi(value.c_str()));
        } else if (arg == "--display-mode") {
            if (!take(options.display_mode)) return false;
            std::transform(options.display_mode.begin(), options.display_mode.end(),
                           options.display_mode.begin(), [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            if (options.display_mode != "none" &&
                options.display_mode != "passthrough" &&
                options.display_mode != "sr") return false;
        } else if (arg == "--display-hidden") {
            options.display_visible = false;
        } else {
            return false;
        }
    }
    return !options.input.empty();
}

bool ReadCompressed(const std::string &path, std::vector<mfxU8> &bytes) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        std::printf("input_open=failed path=%s\n", path.c_str());
        return false;
    }
    const std::streamoff size = stream.tellg();
    if (size <= 0 || static_cast<std::uint64_t>(size) > kMaxCompressedInput) {
        std::printf("input_size=unsupported bytes=%lld max=%zu\n",
                    static_cast<long long>(size), kMaxCompressedInput);
        return false;
    }
    bytes.resize(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char *>(bytes.data()), size);
    return stream.good() || stream.eof();
}

bool CreateDevices(int adapter_index,
                   ComPtr<IDXGIAdapter1> &adapter,
                   ComPtr<ID3D11Device> &d3d11,
                   ComPtr<ID3D11DeviceContext> &d3d11_context,
                   ComPtr<ID3D12Device> &d3d12,
                   Result &result) {
    ComPtr<IDXGIFactory6> factory;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        result.block = "CreateDXGIFactory2=" + HexHr(hr);
        return false;
    }
    hr = factory->EnumAdapters1(static_cast<UINT>(adapter_index), &adapter);
    if (FAILED(hr)) {
        result.block = "EnumAdapters1=" + HexHr(hr);
        return false;
    }
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    char name[256]{};
    std::wcstombs(name, desc.Description, sizeof(name) - 1);
    result.adapter_name = name;
    result.adapter_luid = LuidString(desc.AdapterLuid);

    static const D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0
    };
    D3D_FEATURE_LEVEL selected = D3D_FEATURE_LEVEL_10_0;
    hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                           D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                           levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
                           &d3d11, &selected, &d3d11_context);
    if (FAILED(hr)) {
        result.block = "D3D11CreateDevice=" + HexHr(hr);
        return false;
    }
    ComPtr<IDXGIDevice> dxgi_device;
    if (SUCCEEDED(d3d11.As(&dxgi_device))) {
        ComPtr<IDXGIAdapter> actual_adapter;
        if (SUCCEEDED(dxgi_device->GetAdapter(&actual_adapter))) {
            DXGI_ADAPTER_DESC actual_desc{};
            actual_adapter->GetDesc(&actual_desc);
            result.d3d11_luid = LuidString(actual_desc.AdapterLuid);
        }
    }
    ComPtr<ID3D10Multithread> multithread;
    if (SUCCEEDED(d3d11_context.As(&multithread)))
        multithread->SetMultithreadProtected(TRUE);
    hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                           IID_PPV_ARGS(&d3d12));
    if (SUCCEEDED(hr)) {
        DXGI_ADAPTER_DESC1 d12_desc{};
        adapter->GetDesc1(&d12_desc);
        result.d3d12_luid = LuidString(d12_desc.AdapterLuid);
    } else {
        // D3D12 opening is the proof target, so a missing D3D12 device is a
        // failed bridge, but decode itself may still be reported.
        result.block = "D3D12CreateDevice=" + HexHr(hr);
    }
    std::printf("adapter_name=%s\nadapter_luid=%s\nd3d11_luid=%s\nd3d12_luid=%s\n",
                result.adapter_name.c_str(), result.adapter_luid.c_str(),
                result.d3d11_luid.c_str(), result.d3d12_luid.c_str());
    return true;
}

bool SetFilter(mfxConfig config, const char *name, mfxU32 value) {
    mfxVariant variant{};
    variant.Type = MFX_VARIANT_TYPE_U32;
    variant.Data.U32 = value;
    const mfxStatus status = MFXSetConfigFilterProperty(
        config, reinterpret_cast<mfxU8 *>(const_cast<char *>(name)), variant);
    if (status != MFX_ERR_NONE)
        std::printf("filter=%s status=%s(%d)\n", name, StatusName(status), status);
    return status == MFX_ERR_NONE;
}

bool OpenDecodedTexture(ID3D11Texture2D *texture,
                        ID3D12Device *d3d12,
                        int frame_index,
                        SurfaceLeaseRecord &lease,
                        Result &mutable_result) {
    if (!texture) {
        mutable_result.block = "exported_texture=null";
        return false;
    }
    lease.frame = frame_index;
    lease.exported_lease_acquired = true;
    mutable_result.native_texture_seen++;
    D3D11_TEXTURE2D_DESC d11_desc{};
    texture->GetDesc(&d11_desc);
    lease.source_desc = std::to_string(d11_desc.Width) + "x" +
                        std::to_string(d11_desc.Height) +
                        " format=" + std::to_string(static_cast<unsigned>(d11_desc.Format)) +
                        " array=" + std::to_string(d11_desc.ArraySize) +
                        " mips=" + std::to_string(d11_desc.MipLevels) +
                        " bind=0x" + std::to_string(d11_desc.BindFlags) +
                        " misc=0x" + std::to_string(d11_desc.MiscFlags);
    ComPtr<IDXGIResource1> resource;
    HRESULT hr = texture->QueryInterface(IID_PPV_ARGS(&resource));
    lease.qi_hr = HexHr(hr);
    if (FAILED(hr)) {
        std::printf("frame=%d native=ID3D11Texture2D desc=%ux%u format=%u "
                    "shared_handle=failed hr=%s\n",
                    frame_index, d11_desc.Width, d11_desc.Height,
                    static_cast<unsigned>(d11_desc.Format), HexHr(hr).c_str());
        mutable_result.block = "QueryInterface_IDXGIResource1=" + HexHr(hr);
        return false;
    }
    HANDLE shared_handle = nullptr;
    hr = resource->CreateSharedHandle(nullptr,
                                      DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                      nullptr, &shared_handle);
    lease.create_shared_handle_hr = HexHr(hr);
    if (FAILED(hr) || !shared_handle) {
        std::printf("frame=%d native=ID3D11Texture2D desc=%ux%u format=%u "
                    "shared_handle=failed hr=%s\n",
                    frame_index, d11_desc.Width, d11_desc.Height,
                    static_cast<unsigned>(d11_desc.Format), HexHr(hr).c_str());
        mutable_result.block = "CreateSharedHandle=" + HexHr(hr);
        return false;
    }
    mutable_result.shared_handle_created++;
    if (d3d12) {
        ComPtr<ID3D12Resource> opened;
        hr = d3d12->OpenSharedHandle(shared_handle, IID_PPV_ARGS(&opened));
        lease.open_shared_handle_hr = HexHr(hr);
        if (SUCCEEDED(hr) && opened) {
            const D3D12_RESOURCE_DESC d12_desc = opened->GetDesc();
            lease.opened_desc = std::to_string(static_cast<unsigned long long>(d12_desc.Width)) + "x" +
                                std::to_string(d12_desc.Height) +
                                " format=" + std::to_string(static_cast<unsigned>(d12_desc.Format)) +
                                " array=" + std::to_string(d12_desc.DepthOrArraySize) +
                                " mips=" + std::to_string(d12_desc.MipLevels) +
                                " flags=0x" + std::to_string(static_cast<unsigned>(d12_desc.Flags));
            mutable_result.d3d12_opened++;
            std::printf("frame=%d native=ID3D11Texture2D shared_nt_handle=1 "
                        "d3d12_open=1 width=%llu height=%u format=%u flags=0x%08X "
                        "cpu_map=0 cpu_readback_bytes=0 gpu_copy_count=0\n",
                        frame_index,
                        static_cast<unsigned long long>(d12_desc.Width),
                        d12_desc.Height,
                        static_cast<unsigned>(d12_desc.Format),
                        static_cast<unsigned>(d12_desc.Flags));
        } else {
            std::printf("frame=%d native=ID3D11Texture2D shared_nt_handle=1 "
                        "d3d12_open=0 hr=%s cpu_map=0 cpu_readback_bytes=0\n",
                        frame_index, HexHr(hr).c_str());
            mutable_result.block = "OpenSharedHandle=" + HexHr(hr);
            CloseHandle(shared_handle);
            return false;
        }
    } else {
        std::printf("frame=%d native=ID3D11Texture2D shared_nt_handle=1 "
                    "d3d12_open=blocked cpu_map=0 cpu_readback_bytes=0\n", frame_index);
    }
    CloseHandle(shared_handle);
    return true;
}

void JsonString(std::ofstream &out, const std::string &value) {
    out << '"';
    for (const char c : value) {
        if (c == '\\' || c == '"') out << '\\';
        if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else out << c;
    }
    out << '"';
}

void WriteReport(const std::string &path, const Result &result) {
    if (path.empty()) return;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::printf("report_open=failed path=%s\n", path.c_str());
        return;
    }
    out << "{\n  \"schema_version\": \"full-gpu-pipeline-v2/gate1-decode@1\",\n";
    out << "  \"classification\": ";
    const bool pass = result.requested > 0 && result.decoded == result.requested &&
                      result.decoded == result.exported &&
                      result.exported == result.d3d12_opened &&
                      result.cpu_readback_bytes == 0 && result.cpu_pixel_upload_bytes == 0;
    JsonString(out, pass ? "GPU_RESIDENT_DECODE_SURFACE" : "PARTIAL");
    out << ",\n  \"input\": "; JsonString(out, result.input);
    out << ",\n  \"codec\": "; JsonString(out, result.codec);
    out << ",\n  \"adapter_name\": "; JsonString(out, result.adapter_name);
    out << ",\n  \"adapter_luid\": "; JsonString(out, result.adapter_luid);
    out << ",\n  \"d3d11_luid\": "; JsonString(out, result.d3d11_luid);
    out << ",\n  \"d3d12_luid\": "; JsonString(out, result.d3d12_luid);
    out << ",\n  \"fourcc\": "; JsonString(out, result.fourcc);
    out << ",\n  \"dimensions\": {\"width\": " << result.width
        << ", \"height\": " << result.height
        << ", \"crop_width\": " << result.crop_w
        << ", \"crop_height\": " << result.crop_h << "},\n";
    out << "  \"frames\": {\"requested\": " << result.requested
        << ", \"decoded\": " << result.decoded
        << ", \"exported\": " << result.exported
        << ", \"native_texture_seen\": " << result.native_texture_seen
        << ", \"shared_handle_created\": " << result.shared_handle_created
        << ", \"d3d12_opened\": " << result.d3d12_opened << "},\n";
    out << "  \"surface_lease_records\": [";
    for (std::size_t i = 0; i < result.surface_leases.size(); ++i) {
        const SurfaceLeaseRecord &lease = result.surface_leases[i];
        if (i) out << ",";
        out << "{\"frame\": " << lease.frame
            << ", \"synchronize_status\": "; JsonString(out, lease.synchronize_status);
        out << ", \"export_status\": "; JsonString(out, lease.export_status);
        out << ", \"qi_hr\": "; JsonString(out, lease.qi_hr);
        out << ", \"create_shared_handle_hr\": "; JsonString(out, lease.create_shared_handle_hr);
        out << ", \"open_shared_handle_hr\": "; JsonString(out, lease.open_shared_handle_hr);
        out << ", \"source_desc\": "; JsonString(out, lease.source_desc);
        out << ", \"opened_desc\": "; JsonString(out, lease.opened_desc);
        out << ", \"decoder_lease_acquired\": " << (lease.decoder_lease_acquired ? "true" : "false")
            << ", \"exported_lease_acquired\": " << (lease.exported_lease_acquired ? "true" : "false")
            << ", \"exported_lease_released\": " << (lease.exported_lease_released ? "true" : "false")
            << ", \"decoder_lease_released\": " << (lease.decoder_lease_released ? "true" : "false")
            << "}";
    }
    out << "],\n";
    out << "  \"copy\": {\"d3d12_gpu_copy_count\": " << result.d3d12_copy_count
        << ", \"cpu_pixel_upload_bytes\": " << result.cpu_pixel_upload_bytes
        << ", \"cpu_readback_bytes\": " << result.cpu_readback_bytes << "},\n";
    out << "  \"surface_contract\": {\"native_handle\": \"ID3D11Texture2D\", "
           "\"surface_type\": \"MFX_SURFACE_TYPE_D3D11_TEX2D\", "
           "\"export_flags\": \"MFX_SURFACE_FLAG_EXPORT_SHARED\", "
           "\"cpu_map\": false, \"full_frame_cpu_boundary\": false},\n";
    out << "  \"block\": "; JsonString(out, result.block);
    out << "\n}\n";
}

}  // namespace

int main(int argc, char **argv) {
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        Usage();
        return 2;
    }
    Result result;
    result.input = options.input;
    result.codec = options.codec;
    result.requested = options.max_frames;
    ComRuntime com;
    if (!com.usable()) {
        result.block = "CoInitializeEx=" + HexHr(com.hr);
        WriteReport(options.report, result);
        return 3;
    }
    std::vector<mfxU8> compressed;
    if (!ReadCompressed(options.input, compressed)) return 4;

    const mfxU32 codec = options.codec == "h264" ? MFX_CODEC_AVC : MFX_CODEC_HEVC;
    std::printf("probe=gate1_vpl_decode_surface codec=%s max_frames=%d "
                "compressed_bytes=%zu cpu_pixel_upload_bytes=0 cpu_readback_bytes=0\n",
                options.codec.c_str(), options.max_frames, compressed.size());

    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D11Device> d3d11;
    ComPtr<ID3D11DeviceContext> d3d11_context;
    ComPtr<ID3D12Device> d3d12;
    if (!CreateDevices(options.adapter, adapter, d3d11, d3d11_context, d3d12, result)) {
        WriteReport(options.report, result);
        return 5;
    }
    if (result.d3d12_luid.empty() || result.d3d11_luid.empty() ||
        result.d3d11_luid != result.d3d12_luid) {
        result.block = "adapter_luid_mismatch";
        std::printf("same_luid=0\n");
        WriteReport(options.report, result);
        return 6;
    }
    std::printf("same_luid=1\n");

    mfxLoader loader = MFXLoad();
    if (!loader) {
        result.block = "MFXLoad";
        WriteReport(options.report, result);
        return 7;
    }
    mfxConfig configs[5]{};
    bool filters_ok = true;
    for (auto &config : configs) {
        config = MFXCreateConfig(loader);
        filters_ok = filters_ok && config != nullptr;
    }
    filters_ok = filters_ok &&
        SetFilter(configs[0], "mfxImplDescription.Impl", MFX_IMPL_TYPE_HARDWARE) &&
        SetFilter(configs[1], "mfxImplDescription.ApiVersion.Version", kApiVersion) &&
        SetFilter(configs[2], "mfxImplDescription.AccelerationMode", MFX_ACCEL_MODE_VIA_D3D11) &&
        SetFilter(configs[3], "mfxImplDescription.mfxDecoderDescription.decoder.CodecID", codec);
    filters_ok = filters_ok &&
        SetFilter(configs[4], "mfxSurfaceTypesSupported.surftype.SurfaceType",
                  MFX_SURFACE_TYPE_D3D11_TEX2D) &&
        SetFilter(configs[4], "mfxSurfaceTypesSupported.surftype.surfcomp.SurfaceComponent",
                  MFX_SURFACE_COMPONENT_DECODE) &&
        SetFilter(configs[4], "mfxSurfaceTypesSupported.surftype.surfcomp.SurfaceFlags",
                  MFX_SURFACE_FLAG_EXPORT_SHARED);
    if (!filters_ok) {
        result.block = "MFXSetConfigFilterProperty";
        MFXUnload(loader);
        WriteReport(options.report, result);
        return 8;
    }
    mfxSession session = nullptr;
    mfxStatus status = MFXCreateSession(loader, 0, &session);
    std::printf("MFXCreateSession=%s(%d)\n", StatusName(status), status);
    if (status != MFX_ERR_NONE || !session) {
        result.block = "MFXCreateSession=" + std::to_string(status);
        MFXUnload(loader);
        WriteReport(options.report, result);
        return 9;
    }

    // Bind the native device before querying the implementation or any video
    // component.  Some runtimes lock their default device during a query and
    // then report MFX_ERR_UNDEFINED_BEHAVIOR when a caller device is supplied.
    status = MFXVideoCORE_SetHandle(session, MFX_HANDLE_D3D11_DEVICE,
                                    reinterpret_cast<mfxHDL>(d3d11.Get()));
    std::printf("SetHandle(D3D11)=%s(%d)\n", StatusName(status), status);
    if (status != MFX_ERR_NONE) {
        result.block = "SetHandle=" + std::to_string(status);
        MFXClose(session);
        MFXUnload(loader);
        WriteReport(options.report, result);
        return 10;
    }
    mfxIMPL impl{};
    mfxVersion version{};
    MFXQueryIMPL(session, &impl);
    MFXQueryVersion(session, &version);
    std::printf("onevpl_api=%u.%u impl=0x%08X\n", version.Major, version.Minor,
                static_cast<unsigned>(impl));

    mfxBitstream bitstream{};
    bitstream.Data = compressed.data();
    bitstream.MaxLength = static_cast<mfxU32>(compressed.size());
    bitstream.DataLength = static_cast<mfxU32>(compressed.size());
    bitstream.CodecId = codec;
    mfxVideoParam decode_params{};
    decode_params.mfx.CodecId = codec;
    decode_params.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    status = MFXVideoDECODE_DecodeHeader(session, &bitstream, &decode_params);
    std::printf("DecodeHeader=%s(%d)\n", StatusName(status), status);
    if (status != MFX_ERR_NONE) {
        result.block = "DecodeHeader=" + std::to_string(status);
        MFXClose(session);
        MFXUnload(loader);
        WriteReport(options.report, result);
        return 11;
    }

    status = MFXVideoDECODE_Init(session, &decode_params);
    std::printf("DecodeInit=%s(%d)\n", StatusName(status), status);
    if (status < MFX_ERR_NONE) {
        result.block = "DecodeInit=" + std::to_string(status);
        MFXClose(session);
        MFXUnload(loader);
        WriteReport(options.report, result);
        return 12;
    }
    result.width = decode_params.mfx.FrameInfo.Width;
    result.height = decode_params.mfx.FrameInfo.Height;
    result.crop_w = decode_params.mfx.FrameInfo.CropW;
    result.crop_h = decode_params.mfx.FrameInfo.CropH;
    result.fourcc = FourCCName(decode_params.mfx.FrameInfo.FourCC);
    std::printf("decode_surface=GPU owner=D3D11 native=ID3D11Texture2D "
                "fourcc=%s width=%u height=%u crop=%ux%u cpu_map=0\n",
                result.fourcc.c_str(), result.width, result.height,
                result.crop_w, result.crop_h);

    bool draining = false;
    bool done = false;
    while (!done && result.decoded < options.max_frames) {
        mfxFrameSurface1 *surface = nullptr;
        mfxSyncPoint sync = nullptr;
        const mfxBitstream *input = (!draining && bitstream.DataLength > 0) ? &bitstream : nullptr;
        if (!input) draining = true;
        status = MFXVideoDECODE_DecodeFrameAsync(session,
                                                  const_cast<mfxBitstream *>(input),
                                                  nullptr, &surface, &sync);
        if (status == MFX_ERR_NONE && surface) {
            SurfaceLeaseRecord lease;
            lease.frame = result.decoded;
            lease.decoder_lease_acquired = true;
            status = surface->FrameInterface->Synchronize(surface, 5000);
            lease.synchronize_status = StatusName(status);
            if (status == MFX_ERR_NONE) {
                mfxSurfaceHeader header{};
                header.SurfaceType = MFX_SURFACE_TYPE_D3D11_TEX2D;
                header.SurfaceFlags = MFX_SURFACE_FLAG_EXPORT_SHARED;
                mfxSurfaceHeader *exported_header = nullptr;
                const mfxStatus export_status = surface->FrameInterface->Export(
                    surface, header, &exported_header);
                lease.export_status = StatusName(export_status);
                std::printf("frame=%d DecodeExport=%s(%d)\n", result.decoded,
                            StatusName(export_status), export_status);
                if (export_status == MFX_ERR_NONE && exported_header) {
                    auto *exported = reinterpret_cast<mfxSurfaceD3D11Tex2D *>(exported_header);
                    result.exported++;
                    OpenDecodedTexture(reinterpret_cast<ID3D11Texture2D *>(exported->texture2D),
                                       d3d12.Get(), result.decoded, lease, result);
                    exported->SurfaceInterface.Release(&exported->SurfaceInterface);
                    lease.exported_lease_released = true;
                } else if (result.block.empty()) {
                    result.block = "FrameInterface.Export=" + std::to_string(export_status);
                }
                result.decoded++;
            } else if (result.block.empty()) {
                result.block = "FrameInterface.Synchronize=" + std::to_string(status);
            }
            surface->FrameInterface->Release(surface);
            lease.decoder_lease_released = true;
            result.surface_leases.push_back(std::move(lease));
        } else if (status == MFX_ERR_MORE_DATA) {
            if (draining || bitstream.DataLength == 0) {
                if (result.block.empty() && result.decoded < options.max_frames)
                    result.block = "input_eof_before_requested_frames";
                done = true;
            }
            else draining = true;
        } else if (status == MFX_WRN_DEVICE_BUSY) {
            continue;
        } else if (status == MFX_ERR_MORE_SURFACE) {
            result.block = "MFX_ERR_MORE_SURFACE";
            done = true;
        } else if (status < MFX_ERR_NONE) {
            result.block = "DecodeFrameAsync=" + std::to_string(status);
            done = true;
        }
    }
    MFXVideoDECODE_Close(session);
    MFXClose(session);
    MFXUnload(loader);
    WriteReport(options.report, result);

    const bool pass = result.requested > 0 && result.decoded == result.requested &&
                      result.decoded == result.exported &&
                      result.exported == result.d3d12_opened &&
                      result.d3d11_luid == result.d3d12_luid &&
                      result.cpu_readback_bytes == 0 && result.cpu_pixel_upload_bytes == 0;
    std::printf("summary requested=%d decoded=%d exported=%d d3d12_opened=%d "
                "same_luid=%d cpu_pixel_upload_bytes=%llu cpu_readback_bytes=%llu "
                "classification=%s block=%s\n",
                result.requested, result.decoded, result.exported, result.d3d12_opened,
                result.d3d11_luid == result.d3d12_luid ? 1 : 0,
                static_cast<unsigned long long>(result.cpu_pixel_upload_bytes),
                static_cast<unsigned long long>(result.cpu_readback_bytes),
                pass ? "GPU_RESIDENT_DECODE_SURFACE" : "PARTIAL",
                result.block.empty() ? "none" : result.block.c_str());
    return pass ? 0 : 13;
}
