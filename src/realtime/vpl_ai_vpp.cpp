#define ONEVPL_EXPERIMENTAL

#include <vpl/mfxdispatcher.h>
#include <vpl/mfxvideo.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <tlhelp32.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {

struct Options {
    enum class Operation { kSuperResolution, kFrameInterpolation, kCombined } operation =
        Operation::kSuperResolution;
    std::string input_path  = "-";
    std::string output_path = "-";
    std::string preset      = "default";
    int algorithm           = 0;
    mfxU16 in_width         = 864;
    mfxU16 in_height        = 480;
    mfxU16 out_width        = 1296;
    mfxU16 out_height       = 720;
    mfxU32 in_fps_num       = 24;
    mfxU32 in_fps_den       = 1;
    mfxU32 out_fps_num      = 24;
    mfxU32 out_fps_den      = 1;
    std::string surface_policy = "auto";
    std::string cancel_file;
    mfxU16 matrix = MFX_TRANSFERMATRIX_UNKNOWN;
    mfxU16 range = MFX_NOMINALRANGE_UNKNOWN;
};

const char* StatusName(mfxStatus status) {
    switch (status) {
        case MFX_ERR_NONE: return "MFX_ERR_NONE";
        case MFX_ERR_UNKNOWN: return "MFX_ERR_UNKNOWN";
        case MFX_ERR_NULL_PTR: return "MFX_ERR_NULL_PTR";
        case MFX_ERR_UNSUPPORTED: return "MFX_ERR_UNSUPPORTED";
        case MFX_ERR_MEMORY_ALLOC: return "MFX_ERR_MEMORY_ALLOC";
        case MFX_ERR_NOT_ENOUGH_BUFFER: return "MFX_ERR_NOT_ENOUGH_BUFFER";
        case MFX_ERR_INVALID_HANDLE: return "MFX_ERR_INVALID_HANDLE";
        case MFX_ERR_NOT_INITIALIZED: return "MFX_ERR_NOT_INITIALIZED";
        case MFX_ERR_NOT_FOUND: return "MFX_ERR_NOT_FOUND";
        case MFX_ERR_MORE_DATA: return "MFX_ERR_MORE_DATA";
        case MFX_ERR_MORE_SURFACE: return "MFX_ERR_MORE_SURFACE";
        case MFX_ERR_ABORTED: return "MFX_ERR_ABORTED";
        case MFX_ERR_DEVICE_LOST: return "MFX_ERR_DEVICE_LOST";
        case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM: return "MFX_ERR_INCOMPATIBLE_VIDEO_PARAM";
        case MFX_ERR_INVALID_VIDEO_PARAM: return "MFX_ERR_INVALID_VIDEO_PARAM";
        case MFX_ERR_DEVICE_FAILED: return "MFX_ERR_DEVICE_FAILED";
        case MFX_WRN_IN_EXECUTION: return "MFX_WRN_IN_EXECUTION";
        case MFX_WRN_DEVICE_BUSY: return "MFX_WRN_DEVICE_BUSY";
        case MFX_WRN_VIDEO_PARAM_CHANGED: return "MFX_WRN_VIDEO_PARAM_CHANGED";
        case MFX_WRN_PARTIAL_ACCELERATION: return "MFX_WRN_PARTIAL_ACCELERATION";
        case MFX_WRN_INCOMPATIBLE_VIDEO_PARAM: return "MFX_WRN_INCOMPATIBLE_VIDEO_PARAM";
        case MFX_WRN_FILTER_SKIPPED: return "MFX_WRN_FILTER_SKIPPED";
        default: return "MFX_STATUS_OTHER";
    }
}

void PrintUsage() {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  vpl-ai-vpp sr --in-width W --in-height H --out-width W --out-height H\n"
                 "                [--fps N] [--preset default|sharpen|artifact]\n"
                 "                [--algorithm 0|1|2] [--input FILE|-] [--output FILE|-]\n"
                 "  vpl-ai-vpp fi --width W --height H --in-fps N --out-fps N\n"
                 "                [--preset default|speed|quality]\n"
                 "                [--in-fps-num N --in-fps-den D --out-fps-num N --out-fps-den D]\n"
                 "                [--surface-policy auto|conservative] [--cancel-file FILE]\n"
                 "                [--input FILE|-] [--output FILE|-]\n"
                 "Input and output pixel format: raw NV12.\n");
    std::fprintf(stderr, "Combined: sr-fi uses SR dimensions plus input/output rational fps.\n"
                         "Color: --matrix unknown|bt709|bt601 --range unknown|limited|full\n");
}

bool ParseUnsigned(const char* text, unsigned long min_value, unsigned long max_value, unsigned long* out) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value < min_value || value > max_value)
        return false;
    *out = value;
    return true;
}

bool ParseOptions(int argc, char** argv, Options* options) {
    if (argc < 2)
        return false;

    const std::string operation = argv[1];
    if (operation == "sr") {
        options->operation = Options::Operation::kSuperResolution;
    }
    else if (operation == "fi") {
        options->operation = Options::Operation::kFrameInterpolation;
        options->out_fps_num = 48;
        options->out_fps_den = 1;
        options->out_width = options->in_width;
        options->out_height = options->in_height;
    }
    else if (operation == "sr-fi") {
        options->operation = Options::Operation::kCombined;
        options->out_fps_num = 48;
    }
    else {
        return false;
    }

    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h")
            return false;
        if (i + 1 >= argc)
            return false;
        const char* value = argv[++i];
        unsigned long number = 0;

        if (key == "--input") {
            options->input_path = value;
        }
        else if (key == "--output") {
            options->output_path = value;
        }
        else if (key == "--preset") {
            options->preset = value;
        }
        else if (key == "--algorithm") {
            if (!ParseUnsigned(value, 0, 2, &number))
                return false;
            options->algorithm = static_cast<int>(number);
        }
        else if (key == "--in-width" || key == "--width") {
            if (!ParseUnsigned(value, 16, 16384, &number))
                return false;
            options->in_width = static_cast<mfxU16>(number);
            if (options->operation == Options::Operation::kFrameInterpolation)
                options->out_width = options->in_width;
        }
        else if (key == "--in-height" || key == "--height") {
            if (!ParseUnsigned(value, 16, 16384, &number))
                return false;
            options->in_height = static_cast<mfxU16>(number);
            if (options->operation == Options::Operation::kFrameInterpolation)
                options->out_height = options->in_height;
        }
        else if (key == "--out-width") {
            if (!ParseUnsigned(value, 16, 16384, &number))
                return false;
            options->out_width = static_cast<mfxU16>(number);
        }
        else if (key == "--out-height") {
            if (!ParseUnsigned(value, 16, 16384, &number))
                return false;
            options->out_height = static_cast<mfxU16>(number);
        }
        else if (key == "--fps") {
            if (!ParseUnsigned(value, 1, 1000, &number))
                return false;
            options->in_fps_num = static_cast<mfxU32>(number);
            options->in_fps_den = 1;
            options->out_fps_num = options->in_fps_num;
            options->out_fps_den = 1;
        }
        else if (key == "--in-fps") {
            if (!ParseUnsigned(value, 1, 1000, &number))
                return false;
            options->in_fps_num = static_cast<mfxU32>(number);
            options->in_fps_den = 1;
        }
        else if (key == "--out-fps") {
            if (!ParseUnsigned(value, 1, 1000, &number))
                return false;
            options->out_fps_num = static_cast<mfxU32>(number);
            options->out_fps_den = 1;
        }
        else if (key == "--in-fps-num") {
            if (!ParseUnsigned(value, 1, 1000000, &number))
                return false;
            options->in_fps_num = static_cast<mfxU32>(number);
        }
        else if (key == "--in-fps-den") {
            if (!ParseUnsigned(value, 1, 1000000, &number))
                return false;
            options->in_fps_den = static_cast<mfxU32>(number);
        }
        else if (key == "--out-fps-num") {
            if (!ParseUnsigned(value, 1, 1000000, &number))
                return false;
            options->out_fps_num = static_cast<mfxU32>(number);
        }
        else if (key == "--out-fps-den") {
            if (!ParseUnsigned(value, 1, 1000000, &number))
                return false;
            options->out_fps_den = static_cast<mfxU32>(number);
        }
        else if (key == "--surface-policy") {
            options->surface_policy = value;
            if (options->surface_policy != "auto" &&
                options->surface_policy != "conservative")
                return false;
        }
        else if (key == "--cancel-file") {
            options->cancel_file = value;
        }
        else if (key == "--matrix") {
            const std::string item = value;
            if (item == "bt709") options->matrix = MFX_TRANSFERMATRIX_BT709;
            else if (item == "bt601") options->matrix = MFX_TRANSFERMATRIX_BT601;
            else if (item != "unknown") return false;
        }
        else if (key == "--range") {
            const std::string item = value;
            if (item == "limited") options->range = MFX_NOMINALRANGE_16_235;
            else if (item == "full") options->range = MFX_NOMINALRANGE_0_255;
            else if (item != "unknown") return false;
        }
        else {
            return false;
        }
    }

    if ((options->in_width & 1u) || (options->in_height & 1u) ||
        (options->out_width & 1u) || (options->out_height & 1u)) {
        std::fprintf(stderr, "NV12 dimensions must be even.\n");
        return false;
    }

    if (options->operation != Options::Operation::kFrameInterpolation) {
        if (options->preset != "default" && options->preset != "sharpen" &&
            options->preset != "artifact")
            return false;
    }
    else {
        if (options->preset != "default" && options->preset != "speed" &&
            options->preset != "quality")
            return false;
        if (static_cast<std::uint64_t>(options->out_fps_num) * options->in_fps_den <=
            static_cast<std::uint64_t>(options->in_fps_num) * options->out_fps_den)
            return false;
    }
    if (!options->in_fps_den || !options->out_fps_den)
        return false;
    if (options->operation == Options::Operation::kSuperResolution) {
        options->out_fps_num = options->in_fps_num;
        options->out_fps_den = options->in_fps_den;
    }
    return true;
}

mfxU16 Align16(mfxU16 value) {
    return static_cast<mfxU16>((value + 15u) & ~15u);
}

void FillFrameInfo(mfxFrameInfo* info,
                   mfxU16 width,
                   mfxU16 height,
                   mfxU32 fps_num,
                   mfxU32 fps_den) {
    std::memset(info, 0, sizeof(*info));
    info->FourCC        = MFX_FOURCC_NV12;
    info->ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    info->PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    info->Width         = Align16(width);
    info->Height        = Align16(height);
    info->CropW         = width;
    info->CropH         = height;
    info->FrameRateExtN = fps_num;
    info->FrameRateExtD = fps_den;
}

bool IsCancelled(const Options& options) {
    if (options.cancel_file.empty())
        return false;
#ifdef _WIN32
    const DWORD attrs = GetFileAttributesA(options.cancel_file.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
    FILE* marker = std::fopen(options.cancel_file.c_str(), "rb");
    if (!marker)
        return false;
    std::fclose(marker);
    return true;
#endif
}

// 1 = complete frame; 0 = clean EOF; -1 = partial input or mapping failure.
int ReadNv12Frame(mfxFrameSurface1* surface, FILE* input) {
    mfxStatus status = surface->FrameInterface->Map(surface, MFX_MAP_WRITE);
    if (status != MFX_ERR_NONE) {
        std::fprintf(stderr, "Input surface Map failed: %s (%d)\n", StatusName(status), status);
        return -1;
    }

    const mfxU16 width  = surface->Info.CropW;
    const mfxU16 height = surface->Info.CropH;
    const mfxU16 pitch  = surface->Data.Pitch;
    bool complete       = true;
    size_t bytes_read = 0;

    for (mfxU16 row = 0; row < height; ++row) {
        const size_t got = std::fread(surface->Data.Y + static_cast<size_t>(row) * pitch,
                       1,
                       width,
                       input);
        bytes_read += got;
        if (got != width) {
            complete = false;
            break;
        }
    }
    if (complete) {
        for (mfxU16 row = 0; row < height / 2; ++row) {
            const size_t got = std::fread(surface->Data.UV + static_cast<size_t>(row) * pitch,
                           1,
                           width,
                           input);
            bytes_read += got;
            if (got != width) {
                complete = false;
                break;
            }
        }
    }

    status = surface->FrameInterface->Unmap(surface);
    if (status != MFX_ERR_NONE) {
        std::fprintf(stderr, "Input surface Unmap failed: %s (%d)\n", StatusName(status), status);
        return -1;
    }
    if (!complete && (bytes_read || std::ferror(input))) {
        std::fprintf(stderr, "Truncated NV12 input frame: %llu bytes; refusing partial EOF.\n",
                     static_cast<unsigned long long>(bytes_read));
        return -1;
    }
    return complete ? 1 : 0;
}

bool WriteNv12Frame(mfxFrameSurface1* surface, FILE* output) {
    mfxStatus status = surface->FrameInterface->Map(surface, MFX_MAP_READ);
    if (status != MFX_ERR_NONE) {
        std::fprintf(stderr, "Output surface Map failed: %s (%d)\n", StatusName(status), status);
        return false;
    }

    const mfxU16 width  = surface->Info.CropW;
    const mfxU16 height = surface->Info.CropH;
    const mfxU16 pitch  = surface->Data.Pitch;
    bool complete       = true;

    for (mfxU16 row = 0; row < height; ++row) {
        if (std::fwrite(surface->Data.Y + static_cast<size_t>(row) * pitch,
                        1,
                        width,
                        output) != width) {
            complete = false;
            break;
        }
    }
    if (complete) {
        for (mfxU16 row = 0; row < height / 2; ++row) {
            if (std::fwrite(surface->Data.UV + static_cast<size_t>(row) * pitch,
                            1,
                            width,
                            output) != width) {
                complete = false;
                break;
            }
        }
    }

    status = surface->FrameInterface->Unmap(surface);
    if (status != MFX_ERR_NONE) {
        std::fprintf(stderr, "Output surface Unmap failed: %s (%d)\n", StatusName(status), status);
        return false;
    }
    return complete;
}

mfxStatus SynchronizeSurface(mfxFrameSurface1* surface) {
    for (;;) {
        const mfxStatus status = surface->FrameInterface->Synchronize(surface, 1000);
        if (status == MFX_WRN_IN_EXECUTION)
            continue;
        return status;
    }
}

FILE* OpenInput(const std::string& path) {
    if (path == "-") {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        return stdin;
    }
    FILE* file = nullptr;
#ifdef _WIN32
    fopen_s(&file, path.c_str(), "rb");
#else
    file = std::fopen(path.c_str(), "rb");
#endif
    return file;
}

FILE* OpenOutput(const std::string& path) {
    if (path == "-") {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        return stdout;
    }
    FILE* file = nullptr;
#ifdef _WIN32
    fopen_s(&file, path.c_str(), "wb");
#else
    file = std::fopen(path.c_str(), "wb");
#endif
    return file;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!ParseOptions(argc, argv, &options)) {
        PrintUsage();
        return 2;
    }

    FILE* input = OpenInput(options.input_path);
    if (!input) {
        std::fprintf(stderr, "Could not open input: %s\n", options.input_path.c_str());
        return 3;
    }
    FILE* output = OpenOutput(options.output_path);
    if (!output) {
        std::fprintf(stderr, "Could not open output: %s\n", options.output_path.c_str());
        if (input != stdin)
            std::fclose(input);
        return 4;
    }

    mfxLoader loader = MFXLoad();
    if (!loader) {
        std::fprintf(stderr, "MFXLoad failed. Intel VPL dispatcher is unavailable.\n");
        return 5;
    }

    mfxConfig hardware_config = MFXCreateConfig(loader);
    mfxVariant hardware_value{};
    hardware_value.Type     = MFX_VARIANT_TYPE_U32;
    hardware_value.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    mfxStatus status = MFXSetConfigFilterProperty(
        hardware_config,
        reinterpret_cast<mfxU8*>(const_cast<char*>("mfxImplDescription.Impl")),
        hardware_value);
    if (status != MFX_ERR_NONE) {
        std::fprintf(stderr, "Could not select VPL hardware implementation: %s (%d)\n",
                     StatusName(status), status);
        MFXUnload(loader);
        return 6;
    }

    mfxSession session{};
    status = MFXCreateSession(loader, 0, &session);
    if (status != MFX_ERR_NONE) {
        std::fprintf(stderr, "MFXCreateSession failed: %s (%d)\n", StatusName(status), status);
        MFXUnload(loader);
        return 7;
    }

    mfxVersion version{};
    MFXQueryVersion(session, &version);
    std::fprintf(stderr, "Intel VPL runtime %u.%u\n",
                 static_cast<unsigned>(version.Major),
                 static_cast<unsigned>(version.Minor));

    mfxExtVPPAISuperResolution super_resolution{};
    mfxExtVPPAIFrameInterpolation frame_interpolation{};
    mfxExtVPPVideoSignalInfo signal{};
    mfxExtBuffer* extensions[3]{};
    mfxU16 extension_count = 0;
    if (options.operation != Options::Operation::kFrameInterpolation) {
        super_resolution.Header.BufferId = MFX_EXTBUFF_VPP_AI_SUPER_RESOLUTION;
        super_resolution.Header.BufferSz = sizeof(super_resolution);
        if (options.preset == "sharpen")
            super_resolution.SRMode = MFX_AI_SUPER_RESOLUTION_MODE_SHARPEN;
        else if (options.preset == "artifact")
            super_resolution.SRMode = MFX_AI_SUPER_RESOLUTION_MODE_ARTIFACTREMOVAL;
        else
            super_resolution.SRMode = MFX_AI_SUPER_RESOLUTION_MODE_DEFAULT;
        super_resolution.SRAlgorithm =
            static_cast<mfxAISuperResolutionAlgorithm>(options.algorithm);
        extensions[extension_count++] = reinterpret_cast<mfxExtBuffer*>(&super_resolution);
    }
    if (options.operation != Options::Operation::kSuperResolution) {
        frame_interpolation.Header.BufferId = MFX_EXTBUFF_VPP_AI_FRAME_INTERPOLATION;
        frame_interpolation.Header.BufferSz = sizeof(frame_interpolation);
        frame_interpolation.EnableScd        = 1;
        if (options.preset == "speed")
            frame_interpolation.FIMode = MFX_AI_FRAME_INTERPOLATION_MODE_BEST_SPEED;
        else if (options.preset == "quality")
            frame_interpolation.FIMode = MFX_AI_FRAME_INTERPOLATION_MODE_BEST_QUALITY;
        else
            frame_interpolation.FIMode = MFX_AI_FRAME_INTERPOLATION_MODE_DEFAULT;
        extensions[extension_count++] = reinterpret_cast<mfxExtBuffer*>(&frame_interpolation);
    }
    mfxVideoParam params{};
    if (options.matrix || options.range) {
        signal.Header.BufferId = MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO;
        signal.Header.BufferSz = sizeof(signal);
        signal.In.TransferMatrix = signal.Out.TransferMatrix = options.matrix;
        signal.In.NominalRange = signal.Out.NominalRange = options.range;
        extensions[extension_count++] = reinterpret_cast<mfxExtBuffer*>(&signal);
    }
    FillFrameInfo(&params.vpp.In, options.in_width, options.in_height,
                  options.in_fps_num, options.in_fps_den);
    FillFrameInfo(&params.vpp.Out, options.out_width, options.out_height,
                  options.out_fps_num, options.out_fps_den);
    params.IOPattern   = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    // The implementation owns the VPP surface pool.  AsyncDepth is the
    // supported control that bounds how many surfaces it may retain: keep a
    // small deterministic pool for low-memory systems, while auto permits
    // the driver's normal queue depth without allocating unbounded frames.
    params.AsyncDepth  = options.surface_policy == "conservative" ? 1 : 4;
    params.ExtParam    = extensions;
    params.NumExtParam = extension_count;

    mfxVideoParam queried = params;
    status = MFXVideoVPP_Query(session, &params, &queried);
    std::fprintf(stderr, "AI VPP Query: %s (%d); extensions=%u\n", StatusName(status), status, extension_count);
    if (status == MFX_WRN_FILTER_SKIPPED) {
        std::fprintf(stderr,
                     "AI VPP Query skipped the requested AI filter; refusing silent resize/copy fallback.\n");
        MFXClose(session);
        MFXUnload(loader);
        return 8;
    }
    if (status != MFX_ERR_NONE) {
        std::fprintf(stderr, "AI VPP Query failed: %s (%d)\n", StatusName(status), status);
        MFXClose(session);
        MFXUnload(loader);
        return 8;
    }

    status = MFXVideoVPP_Init(session, &params);
    std::fprintf(stderr, "AI VPP Init: %s (%d); extensions=%u\n", StatusName(status), status, extension_count);
    if (status == MFX_WRN_FILTER_SKIPPED) {
        std::fprintf(stderr,
                     "AI VPP Init skipped the requested AI filter; refusing silent resize/copy fallback.\n");
        MFXVideoVPP_Close(session);
        MFXClose(session);
        MFXUnload(loader);
        return 9;
    }
    if (status != MFX_ERR_NONE) {
        std::fprintf(stderr, "AI VPP Init failed: %s (%d)\n", StatusName(status), status);
        MFXClose(session);
        MFXUnload(loader);
        return 9;
    }

    std::fprintf(stderr,
                 "backend=intel-vpl-ai; %s: NV12 %ux%u@%u/%u -> %ux%u@%u/%u, preset=%s, algorithm=%d, surface-policy=%s\n",
                 options.operation == Options::Operation::kSuperResolution ? "AI SR" :
                 options.operation == Options::Operation::kCombined ? "AI SR+FI" : "AI FI",
                 options.in_width,
                 options.in_height,
                 options.in_fps_num,
                 options.in_fps_den,
                 options.out_width,
                 options.out_height,
                 options.out_fps_num,
                 options.out_fps_den,
                 options.preset.c_str(),
                 options.algorithm,
                 options.surface_policy.c_str());
    std::fprintf(stderr, "vpp_color_matrix=%u; vpp_nominal_range=%u; signal_extension=%u\n",
                 options.matrix, options.range, signal.Header.BufferId ? 1u : 0u);

    const auto start = std::chrono::steady_clock::now();
    mfxFrameSurface1* input_surface = nullptr;
    bool draining = false;
    bool finished = false;
    bool failed   = false;
    std::uint64_t input_frames  = 0;
    std::uint64_t output_frames = 0;

    while (!finished && !failed) {
        if (IsCancelled(options)) {
            std::fprintf(stderr, "cancel requested; draining is aborted\n");
            failed = true;
            break;
        }
        if (!draining && !input_surface) {
            status = MFXMemory_GetSurfaceForVPPIn(session, &input_surface);
            if (status != MFX_ERR_NONE || !input_surface) {
                std::fprintf(stderr, "GetSurfaceForVPPIn failed: %s (%d)\n",
                             StatusName(status), status);
                failed = true;
                break;
            }

            const int read_status = ReadNv12Frame(input_surface, input);
            if (read_status <= 0) {
                input_surface->FrameInterface->Release(input_surface);
                input_surface = nullptr;
                draining      = true;
                if (read_status < 0) {
                    failed = true;
                    break;
                }
            }
            else {
                input_surface->Data.TimeStamp = static_cast<mfxU64>(
                    input_frames * 90000ull * options.in_fps_den / options.in_fps_num);
                ++input_frames;
            }
        }

        mfxFrameSurface1* output_surface = nullptr;
        status = MFXMemory_GetSurfaceForVPPOut(session, &output_surface);
        if (status != MFX_ERR_NONE || !output_surface) {
            std::fprintf(stderr, "GetSurfaceForVPPOut failed: %s (%d)\n",
                         StatusName(status), status);
            failed = true;
            break;
        }

        mfxSyncPoint sync_point{};
        do {
            if (IsCancelled(options)) {
                std::fprintf(stderr, "cancel requested while running AI VPP\n");
                failed = true;
                break;
            }
            status = MFXVideoVPP_RunFrameVPPAsync(session,
                                                   draining ? nullptr : input_surface,
                                                   output_surface,
                                                   nullptr,
                                                   &sync_point);
            if (status == MFX_WRN_DEVICE_BUSY)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (status == MFX_WRN_DEVICE_BUSY);

        if (failed) {
            output_surface->FrameInterface->Release(output_surface);
            break;
        }

        if (status == MFX_WRN_FILTER_SKIPPED) {
            std::fprintf(stderr,
                         "AI VPP skipped its AI filter at runtime; refusing resize/copy fallback\n");
            failed = true;
        }
        const bool produced_output = !failed &&
            (status == MFX_ERR_NONE || status == MFX_ERR_MORE_SURFACE);
        if (produced_output) {
            const mfxStatus sync_status = SynchronizeSurface(output_surface);
            if (sync_status != MFX_ERR_NONE) {
                std::fprintf(stderr, "Output synchronize failed: %s (%d)\n",
                             StatusName(sync_status), sync_status);
                failed = true;
            }
            else if (!WriteNv12Frame(output_surface, output)) {
                std::fprintf(stderr, "Output write failed.\n");
                failed = true;
            }
            else {
                ++output_frames;
            }
        }

        output_surface->FrameInterface->Release(output_surface);

        if (status == MFX_ERR_MORE_SURFACE)
            continue;

        if (input_surface) {
            input_surface->FrameInterface->Release(input_surface);
            input_surface = nullptr;
        }

        if (status == MFX_ERR_MORE_DATA) {
            if (draining)
                finished = true;
            continue;
        }

        if (status < MFX_ERR_NONE) {
            std::fprintf(stderr, "RunFrameVPPAsync failed: %s (%d)\n", StatusName(status), status);
            failed = true;
        }
    }

    if (input_surface)
        input_surface->FrameInterface->Release(input_surface);

    std::fflush(output);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr,
                 "Processed %llu input frames -> %llu output frames in %.3f s (%.2f output fps)\n",
                 static_cast<unsigned long long>(input_frames),
                 static_cast<unsigned long long>(output_frames),
                 elapsed,
                 elapsed > 0.0 ? static_cast<double>(output_frames) / elapsed : 0.0);
    std::fprintf(stderr, "cpu_nv12_upload_bytes=%llu; cpu_nv12_readback_bytes=%llu\n",
        static_cast<unsigned long long>(input_frames * options.in_width * options.in_height * 3ull / 2),
        static_cast<unsigned long long>(output_frames * options.out_width * options.out_height * 3ull / 2));
#ifdef _WIN32
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (snapshot != INVALID_HANDLE_VALUE) {
        MODULEENTRY32W module{};
        module.dwSize = sizeof(module);
        if (Module32FirstW(snapshot, &module)) {
            do {
                std::wstring name = module.szModule;
                if (name.find(L"vpl") != std::wstring::npos ||
                    name.find(L"mfx") != std::wstring::npos ||
                    name.find(L"igd") != std::wstring::npos ||
                    name.find(L"igc") != std::wstring::npos ||
                    name.find(L"openvino") != std::wstring::npos) {
                    char path[4096]{};
                    WideCharToMultiByte(CP_UTF8, 0, module.szExePath, -1, path,
                                        sizeof(path), nullptr, nullptr);
                    std::fprintf(stderr, "runtime_module=%s\n", path);
                }
            } while (Module32NextW(snapshot, &module));
        }
        CloseHandle(snapshot);
    }
#endif

    MFXVideoVPP_Close(session);
    MFXClose(session);
    MFXUnload(loader);

    if (input != stdin)
        std::fclose(input);
    if (output != stdout)
        std::fclose(output);

    return failed ? 10 : 0;
}
