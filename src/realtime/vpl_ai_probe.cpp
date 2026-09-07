#define ONEVPL_EXPERIMENTAL

#include <vpl/mfxdispatcher.h>
#include <vpl/mfxvideo.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

const char* StatusName(mfxStatus status) {
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
void PrintStatus(const char* operation, mfxStatus status) {
    std::printf("%-44s %s (%d)\n", operation, StatusName(status), static_cast<int>(status));
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
    info->FourCC          = MFX_FOURCC_NV12;
    info->ChromaFormat    = MFX_CHROMAFORMAT_YUV420;
    info->PicStruct       = MFX_PICSTRUCT_PROGRESSIVE;
    info->Width           = Align16(width);
    info->Height          = Align16(height);
    info->CropW           = width;
    info->CropH           = height;
    info->FrameRateExtN   = fps_num;
    info->FrameRateExtD   = fps_den;
}

mfxVideoParam MakeVppParams(mfxU16 in_width,
                            mfxU16 in_height,
                            mfxU16 out_width,
                            mfxU16 out_height,
                            mfxU32 in_fps,
                            mfxU32 out_fps,
                            mfxU16 io_pattern,
                            mfxExtBuffer** ext_params,
                            mfxU16 ext_count) {
    mfxVideoParam params{};
    FillFrameInfo(&params.vpp.In, in_width, in_height, in_fps, 1);
    FillFrameInfo(&params.vpp.Out, out_width, out_height, out_fps, 1);
    params.IOPattern   = io_pattern;
    params.ExtParam    = ext_params;
    params.NumExtParam = ext_count;
    return params;
}

void QueryAndInit(mfxSession session, const char* label, mfxVideoParam* params) {
    mfxVideoParam queried = *params;
    mfxStatus status      = MFXVideoVPP_Query(session, params, &queried);

    char operation[160]{};
    std::snprintf(operation, sizeof(operation), "%s Query", label);
    PrintStatus(operation, status);
    if (status < MFX_ERR_NONE)
        return;

    status = MFXVideoVPP_Init(session, params);
    std::snprintf(operation, sizeof(operation), "%s Init", label);
    PrintStatus(operation, status);
    if (status >= MFX_ERR_NONE)
        MFXVideoVPP_Close(session);
}

void ProbeSuperResolution(mfxSession session, mfxU16 io_pattern, const char* memory_label) {
    struct SrCase {
        const char* label;
        mfxAISuperResolutionMode mode;
        mfxAISuperResolutionAlgorithm algorithm;
    } cases[] = {
        { "default", MFX_AI_SUPER_RESOLUTION_MODE_DEFAULT,
          MFX_AI_SUPER_RESOLUTION_ALGORITHM_DEFAULT },
        { "sharpen-alg2", MFX_AI_SUPER_RESOLUTION_MODE_SHARPEN,
          MFX_AI_SUPER_RESOLUTION_ALGORITHM_2 },
        { "artifact-alg2", MFX_AI_SUPER_RESOLUTION_MODE_ARTIFACTREMOVAL,
          MFX_AI_SUPER_RESOLUTION_ALGORITHM_2 },
    };

    for (const SrCase& test : cases) {
        mfxExtVPPAISuperResolution sr{};
        sr.Header.BufferId = MFX_EXTBUFF_VPP_AI_SUPER_RESOLUTION;
        sr.Header.BufferSz = sizeof(sr);
        sr.SRMode           = test.mode;
        sr.SRAlgorithm      = test.algorithm;

        mfxExtBuffer* ext[] = { reinterpret_cast<mfxExtBuffer*>(&sr) };
        mfxVideoParam params = MakeVppParams(864,
                                             480,
                                             1296,
                                             720,
                                             24,
                                             24,
                                             io_pattern,
                                             ext,
                                             1);

        char label[128]{};
        std::snprintf(label, sizeof(label), "AI SR %-13s [%s]", test.label, memory_label);
        QueryAndInit(session, label, &params);
    }
}

void ProbeFrameInterpolation(mfxSession session,
                             mfxU16 io_pattern,
                             const char* memory_label) {
    struct FiCase {
        const char* label;
        mfxAIFrameInterpolationMode mode;
    } cases[] = {
        { "default", MFX_AI_FRAME_INTERPOLATION_MODE_DEFAULT },
        { "best-speed", MFX_AI_FRAME_INTERPOLATION_MODE_BEST_SPEED },
        { "best-quality", MFX_AI_FRAME_INTERPOLATION_MODE_BEST_QUALITY },
    };

    for (const FiCase& test : cases) {
        mfxExtVPPAIFrameInterpolation fi{};
        fi.Header.BufferId = MFX_EXTBUFF_VPP_AI_FRAME_INTERPOLATION;
        fi.Header.BufferSz = sizeof(fi);
        fi.FIMode           = test.mode;
        fi.EnableScd        = 1;

        mfxExtBuffer* ext[] = { reinterpret_cast<mfxExtBuffer*>(&fi) };
        mfxVideoParam params = MakeVppParams(1296,
                                             720,
                                             1296,
                                             720,
                                             24,
                                             48,
                                             io_pattern,
                                             ext,
                                             1);

        char label[128]{};
        std::snprintf(label, sizeof(label), "AI FI %-13s [%s]", test.label, memory_label);
        QueryAndInit(session, label, &params);
    }
}

}  // namespace

int main() {
    std::printf("Intel VPL AI capability probe\n");
    std::printf("Test formats: NV12 864x480 -> 1296x720, 24 -> 48 fps\n\n");

    mfxLoader loader = MFXLoad();
    if (!loader) {
        std::fprintf(stderr, "MFXLoad failed\n");
        return 2;
    }

    mfxConfig hardware_config = MFXCreateConfig(loader);
    if (!hardware_config) {
        std::fprintf(stderr, "MFXCreateConfig failed\n");
        MFXUnload(loader);
        return 3;
    }

    mfxVariant hardware_value{};
    hardware_value.Type     = MFX_VARIANT_TYPE_U32;
    hardware_value.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    mfxStatus status = MFXSetConfigFilterProperty(
        hardware_config,
        reinterpret_cast<mfxU8*>(const_cast<char*>("mfxImplDescription.Impl")),
        hardware_value);
    PrintStatus("Select hardware implementation", status);

    mfxSession session{};
    status = MFXCreateSession(loader, 0, &session);
    PrintStatus("MFXCreateSession", status);
    if (status != MFX_ERR_NONE) {
        MFXUnload(loader);
        return 4;
    }

    mfxVersion version{};
    status = MFXQueryVersion(session, &version);
    PrintStatus("MFXQueryVersion", status);
    if (status == MFX_ERR_NONE)
        std::printf("Runtime API version                          %u.%u\n",
                    static_cast<unsigned>(version.Major),
                    static_cast<unsigned>(version.Minor));

    mfxIMPL implementation{};
    status = MFXQueryIMPL(session, &implementation);
    PrintStatus("MFXQueryIMPL", status);
    if (status == MFX_ERR_NONE)
        std::printf("Implementation flags                         0x%08x\n\n",
                    static_cast<unsigned>(implementation));

    const mfxU16 system_memory =
        MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    const mfxU16 video_memory =
        MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    ProbeSuperResolution(session, system_memory, "system");
    ProbeSuperResolution(session, video_memory, "video");
    std::printf("\n");
    ProbeFrameInterpolation(session, system_memory, "system");
    ProbeFrameInterpolation(session, video_memory, "video");

    MFXClose(session);
    MFXUnload(loader);
    return 0;
}
