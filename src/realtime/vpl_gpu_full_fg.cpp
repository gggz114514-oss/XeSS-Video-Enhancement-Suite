// Experimental product path:
// oneVPL D3D11 decode -> shared NV12 -> D3D12 RGB + GPU Block Motion ->
// XeFG swapchain -> D3D12 copy -> D3D11-owned RGBA -> oneVPL VPP/QSV.
//
// Compressed input/output bytes are CPU I/O.  Decoded and generated pixels
// stay GPU-resident by default. Explicit software encoding reads back only
// the terminal RGB frame, with bounded storage and audited byte counters.
#define XESS_FULL_CHAIN_LIBRARY
#include "vpl_gpu_full_chain_probe.cpp"
#define XESS_FG_LIBRARY
#include "../xess_fg.cpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include "native_gpu_depth.h"
#include "native_gpu_scene.h"
#include "native_gpu_post.h"
#include "native_gpu_ingress.h"

#define XESS_FG_DIS_EMBED
#include "gpu_dis_fg_impl.cpp"
#undef XESS_FG_DIS_EMBED
#include "native_gpu_diagnostic.h"
#include "native_gpu_timers.h"

namespace {

// Real oneVPL pool lease exposed as IUnknown for external-frame ownership.
// AddRef on the D3D texture would not prevent the decoder recycling its pixels.
class MfxPoolOwner final : public IUnknown {
    std::atomic<ULONG> references_{1};
    mfxFrameSurface1* surface_;
    ~MfxPoolOwner() {surface_->FrameInterface->Release(surface_);}
public:
    explicit MfxPoolOwner(mfxFrameSurface1* surface):surface_(surface) {surface_->FrameInterface->AddRef(surface_);}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id,void** out) override {
        if(!out)return E_POINTER;*out=nullptr;
        if(id!=__uuidof(IUnknown))return E_NOINTERFACE;*out=this;AddRef();return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override {return ++references_;}
    ULONG STDMETHODCALLTYPE Release() override {const ULONG count=--references_;if(!count)delete this;return count;}
};

// DecodeFrameAsync returns a surface from oneVPL's pool.  The D3D12 shared
// resource opened from that surface does not, by itself, prevent oneVPL from
// recycling the backing D3D11 texture after FrameInterface::Release.  Keep the
// surface lease until the frame slot's GPU work and encode copy are complete;
// this is the lifetime boundary that makes a decoder worker safe.
struct DecodeSurfaceLease {
    mfxFrameSurface1* surface = nullptr;

    DecodeSurfaceLease() = default;
    ~DecodeSurfaceLease() { reset(); }
    DecodeSurfaceLease(const DecodeSurfaceLease&) = delete;
    DecodeSurfaceLease& operator=(const DecodeSurfaceLease&) = delete;
    DecodeSurfaceLease(DecodeSurfaceLease&& other) noexcept
        : surface(other.surface) { other.surface = nullptr; }
    DecodeSurfaceLease& operator=(DecodeSurfaceLease&& other) noexcept {
        if (this != &other) {
            reset();
            surface = other.surface;
            other.surface = nullptr;
        }
        return *this;
    }

    void reset() {
        if (surface && surface->FrameInterface)
            surface->FrameInterface->Release(surface);
        surface = nullptr;
    }
};

struct FullGpuDecoder {
    Options options;
    Result result;
    std::ifstream compressed_stream;
    std::vector<mfxU8> bitstream_storage;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D11Device> d3d11;
    ComPtr<ID3D11DeviceContext> d3d11_context;
    ComPtr<ID3D12Device> unused_d3d12;
    mfxLoader loader = nullptr;
    mfxSession session = nullptr;
    mfxBitstream bitstream{};
    mfxVideoParam decode_params{};
    mfxExtVideoSignalInfo signal_info{};
    mfxU32 codec = MFX_CODEC_AVC;
    bool draining = false;
    bool input_eof = false;
    std::string error;

    ~FullGpuDecoder() {
        if (session) {
            MFXVideoDECODE_Close(session);
            MFXClose(session);
        }
        if (loader) MFXUnload(loader);
    }

    bool refill() {
        if (input_eof) return bitstream.DataLength > 0;
        if (bitstream.DataOffset && bitstream.DataLength) {
            std::memmove(bitstream.Data,
                         bitstream.Data + bitstream.DataOffset,
                         bitstream.DataLength);
        }
        bitstream.DataOffset = 0;
        const mfxU32 available = bitstream.MaxLength - bitstream.DataLength;
        if (available) {
            compressed_stream.read(
                reinterpret_cast<char*>(bitstream.Data + bitstream.DataLength),
                static_cast<std::streamsize>(available));
            bitstream.DataLength +=
                static_cast<mfxU32>(compressed_stream.gcount());
        }
        if (compressed_stream.eof()) {
            if(options.loop_input) {
                compressed_stream.clear();compressed_stream.seekg(0,std::ios::beg);
            } else {
                input_eof = true;
                bitstream.DataFlag |= MFX_BITSTREAM_EOS;
            }
        } else if (!compressed_stream.good()) {
            error = "compressed_stream_read";
            return false;
        }
        return bitstream.DataLength > 0 || input_eof;
    }

    bool initialize(const Options& input_options) {
        options = input_options;
        result.input = options.input;
        result.codec = options.codec;
        result.requested = options.max_frames;
        compressed_stream.open(options.input, std::ios::binary);
        if (!compressed_stream) {
            error = "read_compressed";
            return false;
        }
        if (!CreateDevices(options.adapter, adapter, d3d11, d3d11_context,
                           unused_d3d12, result)) {
            error = result.block;
            return false;
        }
        codec = options.codec == "h264" ? MFX_CODEC_AVC : MFX_CODEC_HEVC;
        loader = MFXLoad();
        if (!loader) { error = "MFXLoad"; return false; }
        mfxConfig configs[4]{};
        for (auto& config : configs) config = MFXCreateConfig(loader);
        if (!SetFilter(configs[0], "mfxImplDescription.Impl",
                       MFX_IMPL_TYPE_HARDWARE) ||
            !SetFilter(configs[1], "mfxImplDescription.ApiVersion.Version",
                       kApiVersion) ||
            !SetFilter(configs[2], "mfxImplDescription.AccelerationMode",
                       MFX_ACCEL_MODE_VIA_D3D11) ||
            !SetFilter(configs[3],
                       "mfxImplDescription.mfxDecoderDescription.decoder.CodecID",
                       codec)) {
            error = "decoder_filters";
            return false;
        }
        if (MFXCreateSession(loader, 0, &session) != MFX_ERR_NONE || !session) {
            error = "decoder_session";
            return false;
        }
        if (MFXVideoCORE_SetHandle(session, MFX_HANDLE_D3D11_DEVICE,
                                   reinterpret_cast<mfxHDL>(d3d11.Get())) !=
            MFX_ERR_NONE) {
            error = "decoder_set_handle";
            return false;
        }
        constexpr mfxU32 kBitstreamBufferBytes = 4u * 1024u * 1024u;
        bitstream_storage.resize(kBitstreamBufferBytes);
        bitstream.Data = bitstream_storage.data();
        bitstream.MaxLength = kBitstreamBufferBytes;
        bitstream.CodecId = codec;
        if (!refill()) return false;
        decode_params.mfx.CodecId = codec;
        decode_params.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
        signal_info.Header.BufferId = MFX_EXTBUFF_VIDEO_SIGNAL_INFO;
        signal_info.Header.BufferSz = sizeof(signal_info);
        mfxExtBuffer* ext[] = {&signal_info.Header};
        decode_params.ExtParam = ext;
        decode_params.NumExtParam = 1;
        for (;;) {
            const mfxStatus header = MFXVideoDECODE_DecodeHeader(
                session, &bitstream, &decode_params);
            if (header == MFX_ERR_NONE) break;
            if (header != MFX_ERR_MORE_DATA || input_eof || !refill()) {
                error = "DecodeHeader=" + std::to_string(header);
                return false;
            }
        }
        // x264 CRF0 can signal High 4:4:4 Predictive even for yuv420p. The
        // tested B580 decoder accepts it but corrupts pixels silently; fail
        // closed instead of letting finite depth/output counters hide damage.
        if(codec==MFX_CODEC_AVC && (decode_params.mfx.CodecProfile&255u)==244u) {
            error="unsupported_high444_predictive_hardware_decode_profile";return false;
        }
        if(decode_params.mfx.FrameInfo.ChromaFormat!=MFX_CHROMAFORMAT_YUV420 ||
           decode_params.mfx.FrameInfo.FourCC!=MFX_FOURCC_NV12) {
            error="native_core_requires_8bit_NV12_decoder_output";return false;
        }
        mfxVideoParam init = decode_params;
        init.ExtParam = nullptr;
        init.NumExtParam = 0;
        const mfxStatus initialized = MFXVideoDECODE_Init(session, &init);
        if (initialized < MFX_ERR_NONE) {
            error = "DecodeInit=" + std::to_string(initialized);
            return false;
        }
        return true;
    }

    int width() const {
        return decode_params.mfx.FrameInfo.CropW
            ? decode_params.mfx.FrameInfo.CropW : decode_params.mfx.FrameInfo.Width;
    }
    int height() const {
        return decode_params.mfx.FrameInfo.CropH
            ? decode_params.mfx.FrameInfo.CropH : decode_params.mfx.FrameInfo.Height;
    }

    // 1 = frame, 0 = EOS, -1 = failure.
    int next(ID3D12Device* target, ComPtr<ID3D12Resource>& imported,
             DecodeSurfaceLease* lease_out = nullptr) {
        imported.Reset();
        if (lease_out) lease_out->reset();
        for (;;) {
            if (!draining && !input_eof && !refill()) return -1;
            mfxFrameSurface1* surface = nullptr;
            mfxSyncPoint sync = nullptr;
            mfxBitstream* input = !draining ? &bitstream : nullptr;
            const mfxStatus status = MFXVideoDECODE_DecodeFrameAsync(
                session, input, nullptr, &surface, &sync);
            if (status == MFX_WRN_DEVICE_BUSY) {
                Sleep(1);
                continue;
            }
            if (status == MFX_ERR_MORE_DATA) {
                if (!input_eof) {
                    if (!refill()) return -1;
                    continue;
                }
                draining = true;
                return 0;
            }
            if (status < MFX_ERR_NONE) {
                error = "DecodeFrameAsync=" + std::to_string(status);
                return -1;
            }
            // Positive statuses are warnings.  A warning may still carry a
            // valid surface and must not make us drop or leak that frame.
            if (!surface) continue;
            const mfxStatus synchronized =
                surface->FrameInterface->Synchronize(surface, 15000);
            if (synchronized != MFX_ERR_NONE) {
                error = "DecodeSynchronize=" + std::to_string(synchronized);
                surface->FrameInterface->Release(surface);
                return -1;
            }
            mfxSurfaceHeader request{};
            request.SurfaceType = MFX_SURFACE_TYPE_D3D11_TEX2D;
            request.SurfaceFlags = MFX_SURFACE_FLAG_EXPORT_SHARED;
            mfxSurfaceHeader* exported_header = nullptr;
            const mfxStatus exported_status = surface->FrameInterface->Export(
                surface, request, &exported_header);
            if (exported_status != MFX_ERR_NONE || !exported_header) {
                error = "DecodeExport=" + std::to_string(exported_status);
                surface->FrameInterface->Release(surface);
                return -1;
            }
            auto* exported =
                reinterpret_cast<mfxSurfaceD3D11Tex2D*>(exported_header);
            auto* texture =
                reinterpret_cast<ID3D11Texture2D*>(exported->texture2D);
            ComPtr<IDXGIResource1> dxgi;
            HANDLE handle = nullptr;
            bool opened = texture &&
                SUCCEEDED(texture->QueryInterface(IID_PPV_ARGS(&dxgi))) &&
                SUCCEEDED(dxgi->CreateSharedHandle(
                    nullptr,
                    DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                    nullptr, &handle)) && handle;
            if (opened)
                opened = SUCCEEDED(target->OpenSharedHandle(
                    handle, IID_PPV_ARGS(&imported))) && imported;
            if (handle) CloseHandle(handle);
            exported->SurfaceInterface.Release(&exported->SurfaceInterface);
            if (!opened) {
                error = "decode_surface_open";
                surface->FrameInterface->Release(surface);
                return -1;
            }
            if (lease_out) {
                lease_out->surface = surface;
            } else {
                surface->FrameInterface->Release(surface);
            }
            ++result.decoded;
            ++result.exported;
            ++result.d3d12_opened;
            return 1;
        }
    }
};

struct FgFrameSlot {
    xess_gpu::ExternalGpuFrame external_input;
    xess_gpu::MotionPacket motion_packet;
    GpuResource lite_previous, lite_current, lite_forward, lite_backward, lite_confidence;
    UINT lite_width = 0, lite_height = 0;
    ComPtr<ID3D12Resource> color;
    ComPtr<ID3D12Resource> luma;
    ComPtr<ID3D12Resource> forward_flow;
    ComPtr<ID3D12Resource> backward_flow;
    ComPtr<ID3D12Resource> confidence;
    ComPtr<ID3D12Resource> repaired_flow;
    ComPtr<ID3D12Resource> velocity;
    // The persistent XeSS result for this slot.  This is the only color
    // source handed to the XeFG proxy and to the QSV copy endpoint; slot.color
    // remains the input-resolution GPU Block surface only.
    ComPtr<ID3D12Resource> sr_output;
    // GPU DIS computes a responsive mask at input resolution. XeFG's public
    // swap-chain API has no mask resource type, so this per-slot resource is
    // retained for lifetime/diagnostic parity but is not tagged as a fake
    // XeFG resource.
    ComPtr<ID3D12Resource> mask;
    // Optional true-depth upload path used only by the fair OpenVINO
    // ablation. Each slot owns its depth texture so in-flight Presents never
    // race a shared constant-depth resource.
    ComPtr<ID3D12Resource> depth;
    ComPtr<ID3D12Resource> depth_upload;
    ComPtr<ID3D12Resource> imported_nv12;
    DecodeSurfaceLease decoded_surface;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    D3D12_RESOURCE_STATES color_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES luma_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES forward_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES backward_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES confidence_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES repaired_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES velocity_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES sr_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES mask_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES depth_state = D3D12_RESOURCE_STATE_COPY_DEST;
    UINT descriptor_base = 0;
    UINT64 work_fence = 0;
    UINT64 reuse_fence = 0;
    UINT motion_frame_id = 0;
    UINT64 motion_prev_pts_90k = 0;
    UINT64 motion_current_pts_90k = 0;
    UINT64 motion_producer_fence = 0;
    xess_gpu::FencePoint provider_completion;
    bool motion_scene_cut = false;
    bool motion_cpu_ready = false;
    bool motion_consumed_by_sr = false;
    bool motion_consumed_by_fg = false;
    bool depth_ready = false;
    bool timing_ready = false;
};

#include "native_sr_effects.h"

struct GpuSrRuntime {
    NativeSrEffects* effects=nullptr;
    std::vector<FgFrameSlot>* effect_slots=nullptr;
    NativeGpuTimers* gpu_timers=nullptr;
    xess_gpu::MotionProvider* motion_provider=nullptr;
    xess_gpu::Counters provider_counters;
    ComPtr<ID3D12Fence> provider_fence;
    UINT64 next_provider_completion=0;
    ComPtr<ID3D12PipelineState> provider_adapter_pso;
    GpuResource mask_descriptor_sentinel;
    float descriptor_sentinel_value=0;
    xess_context_handle_t context = nullptr;
    UINT input_width = 0;
    UINT input_height = 0;
    UINT output_width = 0;
    UINT output_height = 0;
    ComPtr<ID3D12PipelineState> mask_pso;
    ComPtr<ID3D12PipelineState> lite_downsample_pso, lite_upsample_pso;

    ~GpuSrRuntime() {
        if (context) xessDestroyContext(context);
    }

    bool init(ID3D12Device* device, UINT in_w, UINT in_h, UINT out_w,
              UINT out_h, const std::string& quality) {
        input_width = in_w;
        input_height = in_h;
        output_width = out_w;
        output_height = out_h;
        xess_d3d12_init_params_t params{};
        params.outputResolution = {static_cast<uint32_t>(out_w),
                                   static_cast<uint32_t>(out_h)};
        params.qualitySetting = quality == "performance"
            ? XESS_QUALITY_SETTING_PERFORMANCE
            : quality == "balanced" ? XESS_QUALITY_SETTING_BALANCED
            : quality == "quality" ? XESS_QUALITY_SETTING_QUALITY
            : XESS_QUALITY_SETTING_ULTRA_QUALITY;
        params.initFlags = XESS_INIT_FLAG_LDR_INPUT_COLOR |
                           XESS_INIT_FLAG_HIGH_RES_MV |
                           XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK;
        if (xessD3D12CreateContext(device, &context) != XESS_RESULT_SUCCESS ||
            !context ||
            xessD3D12BuildPipelines(context, nullptr, true, params.initFlags) !=
                XESS_RESULT_SUCCESS ||
            xessD3D12Init(context, &params) != XESS_RESULT_SUCCESS ||
            xessSetVelocityScale(context, 1.0f, 1.0f) != XESS_RESULT_SUCCESS ||
            xessSetMaxResponsiveMaskValue(context, 0.8f) != XESS_RESULT_SUCCESS) {
            if (context) {
                xessDestroyContext(context);
                context = nullptr;
            }
            return false;
        }
        std::fprintf(stderr,
                     "[sr] XeSS context ready input=%ux%u output=%ux%u quality=%s\n",
                     in_w, in_h, out_w, out_h, quality.c_str());
        return true;
    }
};

struct GpuMotionCounters {
    UINT64 provider_adapter_dispatch_count=0;
    UINT64 first_frame_self_analysis_count=0;
    UINT64 diagnostic_readback_bytes=0;
    UINT64 external_input_accept_count = 0, pool_lease_release_count = 0, previous_consumer_fence_count = 0;
    UINT64 directional_dispatch_count = 0;
    UINT64 depth_inference_count = 0;
    UINT64 depth_gpu_copy_count = 0;
    UINT64 depth_pack_count = 0;
    double depth_inference_seconds = 0;
    UINT64 flow_pairs_requested = 0;
    UINT64 flow_pairs_computed = 0;
    UINT64 consumed_by_sr = 0;
    UINT64 consumed_by_fg = 0;
    UINT64 gpu_motion_buffer_copies = 0;
    UINT last_frame_id = 0;
    UINT64 last_prev_pts_90k = 0;
    UINT64 last_current_pts_90k = 0;
    UINT64 last_producer_fence = 0;
    bool last_scene_cut = false;
};

struct EncodeCopySlot {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
};

struct AsyncStats {
    UINT scene_metadata_wait_count = 0, scene_cut_count = 0;
    UINT external_reset_count = 0;
    double scene_metadata_wait_s = 0;
    UINT64 terminal_post_dispatch_count = 0;
    UINT64 diagnostic_post_readback_bytes=0;
    UINT slots = 0;
    UINT max_in_flight = 0;
    UINT decoder_queue_peak = 0;
    UINT encoder_queue_peak = 0;
    UINT slot_wait_count = 0;
    double slot_wait_s = 0.0;
    UINT present_required_wait_count = 0;
    double present_required_wait_s = 0.0;
    UINT capture_wait_count = 0;
    double capture_wait_s = 0.0;
    double encode_worker_wait_s = 0.0;
    UINT dis_graph_wait_count = 0;
    double dis_graph_wait_s = 0.0;
    double decode_s = 0.0;
    double record_submit_s = 0.0;
    double present_s = 0.0;
    double copy_submit_s = 0.0;
    double drain_s = 0.0;
    std::string drain_result = "not_started";
};

bool wait_queue_fence(ID3D12Fence* fence, UINT64 value, DWORD timeout_ms,
                      double* waited_s = nullptr) {
    if (!fence || !value || fence->GetCompletedValue() >= value) return true;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    const auto started = std::chrono::steady_clock::now();
    const bool armed = SUCCEEDED(fence->SetEventOnCompletion(value, event));
    const bool completed = armed && WaitForSingleObject(event, timeout_ms) == WAIT_OBJECT_0;
    if (waited_s)
        *waited_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    CloseHandle(event);
    return completed;
}

#include "native_texture_replay.h"

bool init_fg_slots(Runtime& runtime, UINT input_width, UINT input_height,
                   UINT output_width, UINT output_height, UINT count,
                   std::vector<FgFrameSlot>& slots, float motion_scale) {
    const D3D12_RESOURCE_FLAGS uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    slots.resize(count);
    for (UINT index = 0; index < count; ++index) {
        FgFrameSlot& slot = slots[index];
        slot.descriptor_base = index * 144;
        slot.lite_width = std::max(1u, static_cast<UINT>(std::lround(input_width * motion_scale)));
        slot.lite_height = std::max(1u, static_cast<UINT>(std::lround(input_height * motion_scale)));
        if (motion_scale < .999f) {
            std::string error;
            for (auto* r : {&slot.lite_previous, &slot.lite_current, &slot.lite_confidence})
                if (!chain_texture(runtime.device.Get(), slot.lite_width, slot.lite_height, DXGI_FORMAT_R32_FLOAT,
                    uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, *r, error)) return false;
            for (auto* r : {&slot.lite_forward, &slot.lite_backward})
                if (!chain_texture(runtime.device.Get(), slot.lite_width, slot.lite_height, DXGI_FORMAT_R16G16_FLOAT,
                    uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, *r, error)) return false;
        }
        slot.color = create_texture(runtime.device.Get(), input_width, input_height,
                                     DXGI_FORMAT_R8G8B8A8_UNORM,
                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                     L"XeFG async color", uav);
        slot.luma = create_texture(runtime.device.Get(), input_width, input_height,
                                    DXGI_FORMAT_R32_FLOAT,
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                    L"XeFG async luma", uav);
        slot.forward_flow = create_texture(runtime.device.Get(), input_width, input_height,
                                            DXGI_FORMAT_R16G16_FLOAT,
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                            L"XeFG async forward", uav);
        slot.backward_flow = create_texture(runtime.device.Get(), input_width, input_height,
                                             DXGI_FORMAT_R16G16_FLOAT,
                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                             L"XeFG async backward", uav);
        slot.confidence = create_texture(runtime.device.Get(), input_width, input_height,
                                          DXGI_FORMAT_R32_FLOAT,
                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                          L"XeFG async confidence", uav);
        slot.repaired_flow = create_texture(runtime.device.Get(), input_width, input_height,
                                             DXGI_FORMAT_R16G16_FLOAT,
                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                             L"XeFG async repaired", uav);
        slot.velocity = create_texture(runtime.device.Get(), output_width, output_height,
                                       DXGI_FORMAT_R16G16_FLOAT,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                       L"XeFG async velocity", uav);
        slot.sr_output = create_texture(runtime.device.Get(), output_width, output_height,
                                        DXGI_FORMAT_R8G8B8A8_UNORM,
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                        L"XeSS persistent SR output", uav);
        slot.mask = create_texture(runtime.device.Get(), input_width, input_height,
                                   DXGI_FORMAT_R8_UNORM,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   L"XeFG async DIS mask", uav);
        slot.depth = create_texture(runtime.device.Get(), input_width, input_height,
                                    DXGI_FORMAT_R32_FLOAT,
                                    D3D12_RESOURCE_STATE_COPY_DEST,
                                    L"XeFG async OpenVINO depth");
        const UINT depth_pitch = (input_width * sizeof(float) + 255u) & ~255u;
        slot.depth_upload = create_buffer(
            runtime.device.Get(), static_cast<uint64_t>(depth_pitch) * input_height,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ,
            L"XeFG async OpenVINO depth upload");
        if (!slot.color || !slot.luma || !slot.forward_flow ||
            !slot.backward_flow || !slot.confidence || !slot.repaired_flow ||
            !slot.velocity || !slot.sr_output || !slot.mask || !slot.depth || !slot.depth_upload ||
            FAILED(runtime.device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&slot.allocator))) ||
            FAILED(runtime.device->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_DIRECT, slot.allocator.Get(), nullptr,
                IID_PPV_ARGS(&slot.list))))
            return false;
        if (FAILED(slot.list->Close())) return false;
    }
    return true;
}

bool upload_true_depth(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                      FgFrameSlot& slot, UINT width, UINT height,
                      const float* depth) {
    if (!depth || !slot.depth || !slot.depth_upload) return false;
    const UINT pitch = (width * sizeof(float) + 255u) & ~255u;
    void* mapped = nullptr;
    if (FAILED(slot.depth_upload->Map(0, nullptr, &mapped))) return false;
    auto* bytes = static_cast<std::uint8_t*>(mapped);
    for (UINT y = 0; y < height; ++y) {
        std::memcpy(bytes + static_cast<size_t>(y) * pitch,
                    depth + static_cast<size_t>(y) * width,
                    static_cast<size_t>(width) * sizeof(float));
        if (pitch > width * sizeof(float))
            std::memset(bytes + static_cast<size_t>(y) * pitch + width * sizeof(float),
                        0, pitch - width * sizeof(float));
    }
    slot.depth_upload->Unmap(0, nullptr);
    transition(list, slot.depth.Get(), slot.depth_state,
               D3D12_RESOURCE_STATE_COPY_DEST);
    slot.depth_state = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = slot.depth_upload.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = 0;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
    src.PlacedFootprint.Footprint.Width = width;
    src.PlacedFootprint.Footprint.Height = height;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = pitch;
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = slot.depth.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    transition(list, slot.depth.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    slot.depth_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    (void)device;
    return true;
}

void fg_motion_descriptors(Runtime& runtime, UINT base, UINT srv, UINT uav,
                            ID3D12Resource* t0, DXGI_FORMAT f0,
                            ID3D12Resource* t1, DXGI_FORMAT f1,
                            ID3D12Resource* t2, DXGI_FORMAT f2,
                            ID3D12Resource* t3, DXGI_FORMAT f3,
                            ID3D12Resource* t4, DXGI_FORMAT f4,
                            ID3D12Resource* u0, DXGI_FORMAT fu0,
                            ID3D12Resource* u1, DXGI_FORMAT fu1) {
    write_motion_pass_descriptors(runtime, base + srv, base + uav,
                                  t0, f0, t1, f1, t2, f2, t3, f3, t4, f4,
                                  u0, fu0, u1, fu1);
}

bool record_fg_motion(Runtime& runtime, const Args& args, GpuSrRuntime& sr, FgFrameSlot& slot,
                      FgFrameSlot* previous, int frame_index,
                      UINT input_width, UINT input_height,
                      UINT output_width, UINT output_height) {
    ID3D12GraphicsCommandList* list = slot.list.Get();
    const UINT b = slot.descriptor_base;
    if (frame_index > 0 && previous) {
        if (slot.forward_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            transition(list, slot.forward_flow.Get(), slot.forward_state,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            slot.forward_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (slot.backward_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            transition(list, slot.backward_flow.Get(), slot.backward_state,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            slot.backward_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (slot.confidence_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            transition(list, slot.confidence.Get(), slot.confidence_state,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            slot.confidence_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (args.motion_repair != "off" &&
            slot.repaired_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            transition(list, slot.repaired_flow.Get(), slot.repaired_state,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            slot.repaired_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (slot.velocity_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            transition(list, slot.velocity.Get(), slot.velocity_state,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            slot.velocity_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
    }
    ID3D12DescriptorHeap* heaps[] = {runtime.motion_heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(runtime.motion_root.Get());
    auto bind = [&](UINT srv, UINT uav) {
        list->SetComputeRootDescriptorTable(
            0, gpu_descriptor(runtime.motion_heap.Get(), b + srv,
                              runtime.motion_descriptor_stride));
        list->SetComputeRootDescriptorTable(
            1, gpu_descriptor(runtime.motion_heap.Get(), b + uav,
                              runtime.motion_descriptor_stride));
    };
    UINT constants[6] = {input_width, input_height,
                         input_width, input_height, 0u, 0u};
    auto set_constants = [&]() {
        list->SetComputeRoot32BitConstants(2, 6, constants, 0);
    };
    // Fair FG ablation path. Keep the exact same color conversion, slot,
    // swapchain, constant-depth, capture, and QSV endpoint as gpu-block, but
    // replace only the motion field with zero. This intentionally avoids the
    // block/repair/velocity dispatches while preserving the same resource
    // transitions and synchronization contract.
    if (args.zero_mv_motion && frame_index > 0) {
        fg_motion_descriptors(runtime, b, 32, 37,
                              slot.velocity.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.velocity.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.velocity.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.velocity.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.velocity.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.velocity.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.confidence.Get(), DXGI_FORMAT_R32_FLOAT);
        if (slot.velocity_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            transition(list, slot.velocity.Get(), slot.velocity_state,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            slot.velocity_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        const float zero[4] = {0, 0, 0, 0};
        list->ClearUnorderedAccessViewFloat(
            gpu_descriptor(runtime.motion_heap.Get(), b + 37,
                           runtime.motion_descriptor_stride),
            cpu_descriptor(runtime.motion_heap.Get(), b + 37,
                           runtime.motion_descriptor_stride),
            slot.velocity.Get(), zero, 0, nullptr);
        transition(list, slot.velocity.Get(),
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        slot.velocity_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        return true;
    }
    const UINT repair_mode = args.motion_repair == "propagate" ? 1u :
                             args.motion_repair == "refine" ? 2u : 0u;
    if (frame_index == 0) {
        fg_motion_descriptors(runtime, b, 32, 37,
                              slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.velocity.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.confidence.Get(), DXGI_FORMAT_R32_FLOAT);
        const float zero[4] = {0, 0, 0, 0};
        const UINT clear_indices[3] = {b + 56, b + 57, b + 58};
        ID3D12Resource* clear_resources[3] = {slot.forward_flow.Get(), slot.backward_flow.Get(), slot.confidence.Get()};
        const DXGI_FORMAT clear_formats[3] = {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32_FLOAT};
        for (UINT k = 0; k < 3; ++k) {
            create_uav(runtime.device.Get(), runtime.motion_heap.Get(), clear_indices[k],
                runtime.motion_descriptor_stride, clear_resources[k], clear_formats[k]);
            list->ClearUnorderedAccessViewFloat(gpu_descriptor(runtime.motion_heap.Get(), clear_indices[k], runtime.motion_descriptor_stride),
                cpu_descriptor(runtime.motion_heap.Get(), clear_indices[k], runtime.motion_descriptor_stride), clear_resources[k], zero, 0, nullptr);
            transition(list, clear_resources[k], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        slot.forward_state = slot.backward_state = slot.confidence_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        list->ClearUnorderedAccessViewFloat(
            gpu_descriptor(runtime.motion_heap.Get(), b + 37,
                           runtime.motion_descriptor_stride),
            cpu_descriptor(runtime.motion_heap.Get(), b + 37,
                           runtime.motion_descriptor_stride),
            slot.velocity.Get(), zero, 0, nullptr);
        transition(list, slot.velocity.Get(),
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        slot.velocity_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        return true;
    }
    const bool lite = slot.lite_current.resource != nullptr;
    ID3D12Resource* prev_luma = previous->luma.Get();
    ID3D12Resource* cur_luma = slot.luma.Get();
    ID3D12Resource* primary = slot.forward_flow.Get();
    ID3D12Resource* reverse = slot.backward_flow.Get();
    ID3D12Resource* match_conf = slot.confidence.Get();
    UINT match_w = input_width, match_h = input_height;
    if (lite) {
        auto set = [&](GpuResource& r, D3D12_RESOURCE_STATES state) {
            transition(list,r.resource.Get(),r.state,state); r.state=state;
        };
        for (auto* r : {&slot.lite_previous,&slot.lite_current,&slot.lite_forward,&slot.lite_backward,&slot.lite_confidence})
            set(*r,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        fg_motion_descriptors(runtime,b,64,69,
            prev_luma,DXGI_FORMAT_R32_FLOAT,cur_luma,DXGI_FORMAT_R32_FLOAT,
            nullptr,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,
            slot.lite_previous.resource.Get(),DXGI_FORMAT_R32_FLOAT,slot.lite_current.resource.Get(),DXGI_FORMAT_R32_FLOAT);
        list->SetPipelineState(sr.lite_downsample_pso.Get()); bind(64,69);
        UINT down[8] = {slot.lite_width,slot.lite_height,input_width,input_height,0,0,0,0};
        list->SetComputeRoot32BitConstants(2,8,down,0);
        list->Dispatch((slot.lite_width+7)/8,(slot.lite_height+7)/8,1);
        set(slot.lite_previous,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        set(slot.lite_current,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        prev_luma=slot.lite_previous.resource.Get(); cur_luma=slot.lite_current.resource.Get();
        primary=slot.lite_forward.resource.Get(); reverse=slot.lite_backward.resource.Get(); match_conf=slot.lite_confidence.resource.Get();
        match_w=slot.lite_width; match_h=slot.lite_height;
    }
    fg_motion_descriptors(runtime, b, 8, 13,
                          prev_luma, DXGI_FORMAT_R32_FLOAT,
                          cur_luma, DXGI_FORMAT_R32_FLOAT,
                          slot.luma.Get(), DXGI_FORMAT_R32_FLOAT,
                          slot.luma.Get(), DXGI_FORMAT_R32_FLOAT,
                          slot.luma.Get(), DXGI_FORMAT_R32_FLOAT,
                          primary, DXGI_FORMAT_R16G16_FLOAT,
                          match_conf, DXGI_FORMAT_R32_FLOAT);
    fg_motion_descriptors(runtime, b, 16, 21,
                          cur_luma, DXGI_FORMAT_R32_FLOAT,
                          prev_luma, DXGI_FORMAT_R32_FLOAT,
                          previous->luma.Get(), DXGI_FORMAT_R32_FLOAT,
                          previous->luma.Get(), DXGI_FORMAT_R32_FLOAT,
                          previous->luma.Get(), DXGI_FORMAT_R32_FLOAT,
                          reverse, DXGI_FORMAT_R16G16_FLOAT,
                          match_conf, DXGI_FORMAT_R32_FLOAT);
    list->SetPipelineState(runtime.block_motion_pso.Get());
    bind(8, 13);
    constants[4] = 1u;
    constants[0] = constants[2] = match_w; constants[1] = constants[3] = match_h;
    std::memcpy(&constants[5], &args.motion_center_bias, sizeof(float));
    set_constants();
    list->Dispatch((match_w + 7) / 8, (match_h + 7) / 8, 1);
    bind(16, 21);
    constants[4] = 0u;
    set_constants();
    list->Dispatch((match_w + 7) / 8, (match_h + 7) / 8, 1);
    if (lite) {
        for (auto* r : {&slot.lite_forward,&slot.lite_backward,&slot.lite_confidence}) {
            transition(list,r->resource.Get(),r->state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            r->state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        fg_motion_descriptors(runtime,b,72,77,primary,DXGI_FORMAT_R16G16_FLOAT,
            match_conf,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,
            slot.forward_flow.Get(),DXGI_FORMAT_R16G16_FLOAT,slot.confidence.Get(),DXGI_FORMAT_R32_FLOAT);
        fg_motion_descriptors(runtime,b,80,85,reverse,DXGI_FORMAT_R16G16_FLOAT,
            match_conf,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,
            slot.backward_flow.Get(),DXGI_FORMAT_R16G16_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT);
        list->SetPipelineState(sr.lite_upsample_pso.Get());
        UINT up[8]={input_width,input_height,input_width,input_height,match_w,match_h,0,0};
        list->SetComputeRoot32BitConstants(2,8,up,0);
        bind(72,77); list->Dispatch((input_width+7)/8,(input_height+7)/8,1);
        bind(80,85); list->Dispatch((input_width+7)/8,(input_height+7)/8,1);
        constants[0]=constants[2]=input_width; constants[1]=constants[3]=input_height;
    }
    transition(list, slot.forward_flow.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(list, slot.backward_flow.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(list, slot.confidence.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    slot.forward_state = slot.backward_state = slot.confidence_state =
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    ID3D12Resource* selected = slot.forward_flow.Get();
    if (repair_mode != 0u) {
        fg_motion_descriptors(runtime, b, 24, 29,
                              slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.backward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.confidence.Get(), DXGI_FORMAT_R32_FLOAT,
                              slot.luma.Get(), DXGI_FORMAT_R32_FLOAT,
                              previous->luma.Get(), DXGI_FORMAT_R32_FLOAT,
                              slot.repaired_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                              slot.confidence.Get(), DXGI_FORMAT_R32_FLOAT);
        list->SetPipelineState(runtime.repair_pso.Get());
        bind(24, 29);
        constants[4] = repair_mode;
        set_constants();
        list->Dispatch((input_width + 7) / 8, (input_height + 7) / 8, 1);
        transition(list, slot.repaired_flow.Get(),
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        slot.repaired_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        selected = slot.repaired_flow.Get();
    }
    fg_motion_descriptors(runtime, b, 32, 37,
                          selected, DXGI_FORMAT_R16G16_FLOAT,
                          slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                          slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                          slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                          slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
                          slot.velocity.Get(), DXGI_FORMAT_R16G16_FLOAT,
                          slot.confidence.Get(), DXGI_FORMAT_R32_FLOAT);
    list->SetPipelineState(runtime.velocity_pso.Get());
    constants[0] = output_width;
    constants[1] = output_height;
    constants[2] = input_width;
    constants[3] = input_height;
    constants[4] = constants[5] = 0u;
    set_constants();
    list->Dispatch((output_width + 7) / 8, (output_height + 7) / 8, 1);
    transition(list, slot.velocity.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    slot.velocity_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    return true;
}

bool record_decoded_fg_frame(Runtime& runtime, const Args& args,
                             GpuSrRuntime& sr, GpuMotionCounters& counters,
                             FgFrameSlot& slot, FgFrameSlot* previous,
                             ID3D12Resource* imported_nv12,
                             ID3D12Resource* backbuffer, int frame_index,
                             UINT input_width, UINT input_height,
                             UINT output_width, UINT output_height,
                             const ColorSelection& color,
                             const float* true_depth, NativeGpuDepth* gpu_depth,
                             UINT slot_index, int previous_index, bool run_sr,
                             NativeGpuScene& scene) {
    if (!slot.allocator || FAILED(slot.allocator->Reset()) ||
        FAILED(slot.list->Reset(slot.allocator.Get(), nullptr))) return false;
    const UINT b = slot.descriptor_base;
    create_srv(runtime.device.Get(), runtime.motion_heap.Get(), b + 40,
               runtime.motion_descriptor_stride, imported_nv12,
               DXGI_FORMAT_R8_UNORM, 0);
    create_srv(runtime.device.Get(), runtime.motion_heap.Get(), b + 41,
               runtime.motion_descriptor_stride, imported_nv12,
               DXGI_FORMAT_R8G8_UNORM, 1);
    for (UINT i = 42; i <= 44; ++i)
        create_srv(runtime.device.Get(), runtime.motion_heap.Get(), b + i,
                   runtime.motion_descriptor_stride, imported_nv12,
                   DXGI_FORMAT_R8_UNORM, 0);
    create_uav(runtime.device.Get(), runtime.motion_heap.Get(), b + 45,
               runtime.motion_descriptor_stride, slot.color.Get(),
               DXGI_FORMAT_R8G8B8A8_UNORM);
    create_uav(runtime.device.Get(), runtime.motion_heap.Get(), b + 46,
               runtime.motion_descriptor_stride, slot.luma.Get(),
               DXGI_FORMAT_R32_FLOAT);
    ID3D12GraphicsCommandList* list = slot.list.Get();
    transition(list, slot.color.Get(), slot.color_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    slot.color_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if (slot.luma_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        transition(list, slot.luma.Get(), slot.luma_state,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        slot.luma_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    transition(list, imported_nv12, D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ID3D12DescriptorHeap* heaps[] = {runtime.motion_heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(runtime.motion_root.Get());
    list->SetPipelineState(runtime.nv12_color_pso.Get());
    list->SetComputeRootDescriptorTable(
        0, gpu_descriptor(runtime.motion_heap.Get(), b + 40,
                          runtime.motion_descriptor_stride));
    list->SetComputeRootDescriptorTable(
        1, gpu_descriptor(runtime.motion_heap.Get(), b + 45,
                          runtime.motion_descriptor_stride));
    UINT constants[6] = {
        input_width, input_height,
        color.matrix == ColorMatrix::Bt709 ? 1u : 0u,
        color.range == ColorRange::Full ? 1u : 0u, 0u, 0u};
    list->SetComputeRoot32BitConstants(2, 6, constants, 0);
    list->Dispatch((input_width + 7) / 8, (input_height + 7) / 8, 1);
    transition(list, slot.luma.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    slot.luma_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    create_uav(runtime.device.Get(), runtime.motion_heap.Get(), b + 47,
               runtime.motion_descriptor_stride, slot.mask.Get(), DXGI_FORMAT_R8_UNORM);
    if (slot.mask_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        transition(list, slot.mask.Get(), slot.mask_state,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        slot.mask_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    transition(list, imported_nv12,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    if (true_depth && !upload_true_depth(runtime.device.Get(), list, slot,
                                         input_width, input_height, true_depth))
        return false;
    slot.motion_frame_id = static_cast<UINT>(frame_index);
    slot.motion_prev_pts_90k = frame_index > 0
        ? static_cast<UINT64>((frame_index - 1) * 90000.0 / args.fps + 0.5)
        : 0;
    slot.motion_current_pts_90k = static_cast<UINT64>(
        frame_index * 90000.0 / args.fps + 0.5);
    slot.motion_scene_cut = false;
    slot.motion_cpu_ready = false;
    slot.motion_consumed_by_sr = false;
    slot.motion_consumed_by_fg = false;
    if(sr.gpu_timers)sr.gpu_timers->mark(list,slot_index,0);
    if(sr.motion_provider) {
        if(!sr.provider_fence && FAILED(runtime.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&sr.provider_fence))))return false;
        // A distinct retirement timeline cannot be accidentally completed by
        // an unrelated runtime copy signal while the provider is recording.
        slot.provider_completion={sr.provider_fence,++sr.next_provider_completion};
        xess_gpu::RecordContext context{runtime.device.Get(),list,runtime.queue.Get(),runtime.device->GetAdapterLuid(),slot.provider_completion};
        // NV12->RGBA has already been recorded above. This transition orders
        // that UAV producer before optional provider reads, without another
        // color conversion/allocation. Previous-slot lifetime was pinned by its
        // source lease and is extended by this pair's consumer fence below.
        transition(list,slot.color.Get(),slot.color_state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        slot.color_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        xess_gpu::TextureView current_rgba{slot.color,DXGI_FORMAT_R8G8B8A8_UNORM,
            {input_width,input_height},{0,0,input_width,input_height},slot.color_state};
        xess_gpu::TextureView previous_rgba;
        context.current_rgba=&current_rgba;
        if(previous) {
            transition(list,previous->color.Get(),previous->color_state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            previous->color_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            previous_rgba={previous->color,DXGI_FORMAT_R8G8B8A8_UNORM,
                {input_width,input_height},{0,0,input_width,input_height},previous->color_state};
            context.previous_rgba=&previous_rgba;
        }
        const auto before=sr.provider_counters;
        std::string provider_error;
        if(!sr.motion_provider->record(previous?&previous->external_input.frame:nullptr,slot.external_input.frame,
            slot_index,context,slot.motion_packet,sr.provider_counters,provider_error)) {
            std::fprintf(stderr,"[provider] %s\n",provider_error.c_str());return false;
        }
        counters.flow_pairs_requested+=sr.provider_counters.source_pair_analysis_count-before.source_pair_analysis_count;
        counters.flow_pairs_computed+=sr.provider_counters.source_pair_analysis_count-before.source_pair_analysis_count;
        counters.directional_dispatch_count+=sr.provider_counters.directional_dispatch_count-before.directional_dispatch_count;
        counters.first_frame_self_analysis_count+=sr.provider_counters.first_frame_self_analysis_count-before.first_frame_self_analysis_count;
        if(frame_index==0) {
            if(!record_fg_motion(runtime,args,sr,slot,nullptr,0,input_width,input_height,output_width,output_height))return false;
        } else {
            auto& packet=slot.motion_packet;
            if(!xess_gpu::matches(slot.external_input.frame,packet) || !packet.has_reverse ||
               !packet.current_to_previous.resource || !packet.previous_to_current.resource ||
               packet.current_to_previous.format!=DXGI_FORMAT_R32G32_FLOAT || packet.previous_to_current.format!=DXGI_FORMAT_R32G32_FLOAT ||
               packet.current_to_previous.state!=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ||
               packet.previous_to_current.state!=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) {
                std::fprintf(stderr,"[provider] packet must be matched, bidirectional source-sized R32G32_FLOAT readable textures\n");return false;
            }
            ID3D12DescriptorHeap* adapter_heaps[]={runtime.motion_heap.Get()};list->SetDescriptorHeaps(1,adapter_heaps);
            list->SetComputeRootSignature(runtime.motion_root.Get());
            transition(list,slot.forward_flow.Get(),slot.forward_state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition(list,slot.backward_flow.Get(),slot.backward_state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition(list,slot.confidence.Get(),slot.confidence_state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            for(UINT direction=0;direction<2;++direction) {
                const UINT d=direction?120:112;
                fg_motion_descriptors(runtime,b,d,d+5,
                    direction?packet.previous_to_current.resource.Get():packet.current_to_previous.resource.Get(),DXGI_FORMAT_R32G32_FLOAT,
                    direction?packet.current_to_previous.resource.Get():packet.previous_to_current.resource.Get(),DXGI_FORMAT_R32G32_FLOAT,
                    previous->luma.Get(),DXGI_FORMAT_R32_FLOAT,slot.luma.Get(),DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,
                    direction?slot.backward_flow.Get():slot.forward_flow.Get(),DXGI_FORMAT_R16G16_FLOAT,
                    direction?nullptr:slot.confidence.Get(),DXGI_FORMAT_R32_FLOAT);
                list->SetPipelineState(sr.provider_adapter_pso.Get());
                list->SetComputeRootDescriptorTable(0,gpu_descriptor(runtime.motion_heap.Get(),b+d,runtime.motion_descriptor_stride));
                list->SetComputeRootDescriptorTable(1,gpu_descriptor(runtime.motion_heap.Get(),b+d+5,runtime.motion_descriptor_stride));
                UINT c[8]={input_width,input_height,direction,0,0,0,0,0};list->SetComputeRoot32BitConstants(2,8,c,0);
                list->Dispatch((input_width+7)/8,(input_height+7)/8,1);++counters.provider_adapter_dispatch_count;
            }
            transition(list,slot.forward_flow.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition(list,slot.backward_flow.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition(list,slot.confidence.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            slot.forward_state=slot.backward_state=slot.confidence_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            transition(list,slot.velocity.Get(),slot.velocity_state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            fg_motion_descriptors(runtime,b,32,37,slot.forward_flow.Get(),DXGI_FORMAT_R16G16_FLOAT,
                nullptr,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT,
                slot.velocity.Get(),DXGI_FORMAT_R16G16_FLOAT,nullptr,DXGI_FORMAT_R32_FLOAT);
            list->SetPipelineState(runtime.velocity_pso.Get());
            list->SetComputeRootDescriptorTable(0,gpu_descriptor(runtime.motion_heap.Get(),b+32,runtime.motion_descriptor_stride));
            list->SetComputeRootDescriptorTable(1,gpu_descriptor(runtime.motion_heap.Get(),b+37,runtime.motion_descriptor_stride));
            UINT c[8]={output_width,output_height,input_width,input_height,0,0,0,0};list->SetComputeRoot32BitConstants(2,8,c,0);
            list->Dispatch((output_width+7)/8,(output_height+7)/8,1);++counters.provider_adapter_dispatch_count;
            transition(list,slot.velocity.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            slot.velocity_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
    } else if (!record_fg_motion(runtime, args, sr, slot, previous, frame_index,
                          input_width, input_height, output_width, output_height)) return false;
    if(sr.gpu_timers)sr.gpu_timers->mark(list,slot_index,1);
    if (frame_index > 0 && !sr.motion_provider) {
        ++counters.flow_pairs_requested;
        ++counters.flow_pairs_computed;
        counters.directional_dispatch_count += args.motion_repair == "refine" ? 3 : 2;
    }
    // Reuse the two saved source-direction fields. The mask pass performs no
    // matching and uses a unique descriptor range for this in-flight slot.
    fg_motion_descriptors(runtime, b, 48, 53,
        previous ? previous->luma.Get() : slot.luma.Get(), DXGI_FORMAT_R32_FLOAT,
        slot.luma.Get(), DXGI_FORMAT_R32_FLOAT,
        slot.forward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
        slot.backward_flow.Get(), DXGI_FORMAT_R16G16_FLOAT,
        slot.confidence.Get(), DXGI_FORMAT_R32_FLOAT,
        nullptr, DXGI_FORMAT_R32_FLOAT, slot.mask.Get(), DXGI_FORMAT_R8_UNORM);
    list->SetPipelineState(sr.mask_pso.Get());
    list->SetComputeRootDescriptorTable(0, gpu_descriptor(runtime.motion_heap.Get(), b+48, runtime.motion_descriptor_stride));
    list->SetComputeRootDescriptorTable(1, gpu_descriptor(runtime.motion_heap.Get(), b+53, runtime.motion_descriptor_stride));
    UINT mask_constants[6] = {input_width,input_height,input_width,input_height,sr.motion_provider?0u:1u,0};
    if(sr.mask_descriptor_sentinel.resource) {
        create_uav(runtime.device.Get(),runtime.motion_heap.Get(),b+55,runtime.motion_descriptor_stride,
            sr.mask_descriptor_sentinel.resource.Get(),DXGI_FORMAT_R8_UNORM);
        const float value[4]={sr.descriptor_sentinel_value,0,0,0};
        list->ClearUnorderedAccessViewFloat(gpu_descriptor(runtime.motion_heap.Get(),b+55,runtime.motion_descriptor_stride),
            cpu_descriptor(runtime.motion_heap.Get(),b+55,runtime.motion_descriptor_stride),sr.mask_descriptor_sentinel.resource.Get(),value,0,nullptr);
    }
    list->SetComputeRoot32BitConstants(2,6,mask_constants,0);
    list->Dispatch((input_width+7)/8,(input_height+7)/8,1);
    transition(list, slot.mask.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    slot.mask_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if(sr.gpu_timers)sr.gpu_timers->mark(list,slot_index,2);
    if (gpu_depth) {
        gpu_depth->record(slot_index, previous_index, list,
            slot.forward_flow.Get(), slot.confidence.Get(), slot.mask.Get(), frame_index==0);
        slot.depth_ready = true;
    }
    if(sr.gpu_timers)sr.gpu_timers->mark(list,slot_index,3);
    scene.record(runtime.device.Get(),runtime.motion_root.Get(),list,slot_index,
        previous ? previous->luma.Get() : slot.luma.Get(),slot.luma.Get(),slot.mask.Get(),input_width,input_height);
    if(sr.gpu_timers)sr.gpu_timers->mark(list,slot_index,4);

    return true;
}

// Compile-time reusable entry for decoded files or external capture surfaces.
// It owns no decoder and performs no pixel transfers through host memory.
// Define XESS_NATIVE_GPU_CORE_LIBRARY when embedding this translation unit.
bool record_external_gpu_source(Runtime& runtime, const Args& args, GpuSrRuntime& sr,
                                GpuMotionCounters& counters, FgFrameSlot& slot,
                                FgFrameSlot* previous, UINT source_ordinal,
                                UINT input_width, UINT input_height,
                                UINT output_width, UINT output_height,
                                const ColorSelection& color, NativeGpuDepth* depth,
                                UINT slot_index, int previous_index, NativeGpuScene& scene,
                                std::string& error) {
    auto& input=slot.external_input;
    if(!xess_gpu::accept_external_gpu_frame(input,runtime.device.Get(),runtime.queue.Get(),
        {input_width,input_height},error)) return false;
    if(previous && (input.frame.metadata.previous_source_frame_id!=previous->external_input.frame.metadata.source_frame_id ||
       input.frame.metadata.source_frame_id<=previous->external_input.frame.metadata.source_frame_id)) {
        error="external_input_previous_identity_mismatch";return false;
    }
    if(previous && input.frame.metadata.source_frame_id!=previous->external_input.frame.metadata.source_frame_id+1 &&
       input.frame.metadata.reset==xess_gpu::ResetReason::None) {
        error="external_input_gap_requires_explicit_reset";return false;
    }
    if(depth) {
        // D3D12 consumption uses a GPU Wait. OpenVINO's D3D11/CL interop has a
        // different queue; it may only acquire a completed external producer.
        // Record this ingress-specific fence wait, never assume host visibility.
        if(input.frame.produced.value && !wait_queue_fence(input.frame.produced.fence.Get(),input.frame.produced.value,30000)) {
            error="external_depth_producer_fence";return false;
        }
        if(!depth->infer(slot_index,input.nv12_d3d11.Get(),input.frame.metadata,input.frame.color.valid,error)) return false;
    }
    ++counters.external_input_accept_count;
    return record_decoded_fg_frame(runtime,args,sr,counters,slot,previous,input.frame.color.resource.Get(),nullptr,
        source_ordinal,input_width,input_height,output_width,output_height,color,nullptr,depth,slot_index,previous_index,true,scene);
}

bool record_gpu_consumers(Runtime& runtime, GpuSrRuntime& sr, GpuMotionCounters& counters,
                          FgFrameSlot& slot, ID3D12Resource* backbuffer,
                          UINT input_width, UINT input_height, bool reset, bool run_sr) {
    if (FAILED(slot.allocator->Reset()) || FAILED(slot.list->Reset(slot.allocator.Get(), nullptr))) return false;
    ID3D12GraphicsCommandList* list = slot.list.Get();
    const UINT timing_slot=slot.descriptor_base/144;
    if(sr.gpu_timers)sr.gpu_timers->mark(list,timing_slot,5);
    // XeSS consumes the same packet's velocity/confidence-derived SRV that
    // XeFG will tag below.  The result stays in this slot until both the
    // explicit proxy copy and the asynchronous encoder copy have completed.
    transition(list, slot.color.Get(), slot.color_state,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    slot.color_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    transition(list, slot.sr_output.Get(), slot.sr_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    slot.sr_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    xess_d3d12_execute_params_t exec{};
    exec.inputWidth = input_width;
    exec.inputHeight = input_height;
    exec.jitterOffsetX = 0.0f;
    exec.jitterOffsetY = 0.0f;
    exec.exposureScale = 1.0f;
    exec.resetHistory = reset ? 1 : 0;
    exec.pColorTexture = slot.color.Get();
    exec.pVelocityTexture = slot.velocity.Get();
    exec.pOutputTexture = slot.sr_output.Get();
    exec.pDepthTexture = nullptr;
    exec.pExposureScaleTexture = nullptr;
    exec.pResponsivePixelMaskTexture = slot.mask.Get();
    if (run_sr) {
        if (xessD3D12Execute(sr.context, list, &exec) != XESS_RESULT_SUCCESS) return false;
        slot.motion_consumed_by_sr = true;
        ++counters.consumed_by_sr;
        if(sr.effects)sr.effects->record(runtime.device.Get(),list,*sr.effect_slots,timing_slot,reset);
    } else {
        transition(list, slot.color.Get(), slot.color_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        slot.color_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
        transition(list, slot.sr_output.Get(), slot.sr_state, D3D12_RESOURCE_STATE_COPY_DEST);
        slot.sr_state = D3D12_RESOURCE_STATE_COPY_DEST;
        list->CopyResource(slot.sr_output.Get(), slot.color.Get());
    }
    transition(list, slot.sr_output.Get(), slot.sr_state,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    slot.sr_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    if(sr.gpu_timers) {sr.gpu_timers->mark(list,timing_slot,6);sr.gpu_timers->resolve(list,timing_slot);slot.timing_ready=true;}
    transition(list, backbuffer, D3D12_RESOURCE_STATE_PRESENT,
               D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyResource(backbuffer, slot.sr_output.Get());
    ++counters.gpu_motion_buffer_copies;
    transition(list, slot.sr_output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    slot.sr_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    transition(list, backbuffer, D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return true;
}

// GPU DIS variant of the frame recorder.  The imported D3D11 NV12 surface is
// still consumed directly by D3D12; the DIS implementation owns one
// persistent A/B pyramid and scratch graph.  Its output is copied into the
// frame slot before the command list is submitted, so the shared graph can be
// safely reused only after this submission's fence.  This is the explicit
// serial island imposed by DIS's persistent pyramid, while decode, capture,
// and QSV copy/encode remain in the bounded outer pipeline.
bool record_decoded_gpu_dis_frame(Runtime& runtime, const Args& args,
                                  FgFrameSlot& slot, DisPipeline& dis,
                                  ID3D12Resource* imported_nv12,
                                  ID3D12Resource* backbuffer,
                                  int frame_index,
                                  const ColorSelection& color) {
    if (!slot.allocator || FAILED(slot.allocator->Reset()) ||
        FAILED(slot.list->Reset(slot.allocator.Get(), nullptr))) return false;
    ID3D12GraphicsCommandList* list = slot.list.Get();
    transition(list, imported_nv12, D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    const UINT role = (static_cast<UINT>(frame_index) & 1u) ? 1u : 0u;
    const bool frame0_self = frame_index == 0;
    R& luma_cur = (role == 0) ? dis.luma_a : dis.luma_b;
    R& luma_prev = (role == 0) ? dis.luma_b : dis.luma_a;
    dis.bind_role_descriptors(role, frame0_self, imported_nv12);
    dis.record_color(list, imported_nv12, luma_cur,
                     color.matrix == ColorMatrix::Bt709 ? 1u : 0u,
                     color.range == ColorRange::Full ? 1u : 0u);
    dis.record_graph(list, role, frame0_self);
    dis.record_bridge(list, false, false);
    // XeFG has no public responsive-mask resource enum.  Still execute the
    // strict GPU DIS mask graph into a per-slot copy so this backend's motion
    // and mask work has the same lifetime and observability as the archive.
    dis.record_mask(list, frame0_self ? luma_cur.p.Get() : luma_prev.p.Get(),
                    luma_cur.p.Get());

    auto copy_dis = [&](R& source, ID3D12Resource* destination,
                        D3D12_RESOURCE_STATES& destination_state,
                        const wchar_t* label) {
        (void)label;
        transition(list, source.p.Get(), source.s,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        source.s = D3D12_RESOURCE_STATE_COPY_SOURCE;
        transition(list, destination, destination_state,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        destination_state = D3D12_RESOURCE_STATE_COPY_DEST;
        list->CopyResource(destination, source.p.Get());
        transition(list, destination, D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        destination_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        transition(list, source.p.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        source.s = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        ++dis.product_gpu_copies;
    };
    copy_dis(dis.color, slot.color.Get(), slot.color_state, L"color");
    copy_dis(dis.velocity, slot.velocity.Get(), slot.velocity_state, L"velocity");
    copy_dis(dis.mask, slot.mask.Get(), slot.mask_state, L"mask");

    transition(list, slot.color.Get(), slot.color_state,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    slot.color_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    transition(list, backbuffer, D3D12_RESOURCE_STATE_PRESENT,
               D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyResource(backbuffer, slot.color.Get());
    transition(list, slot.color.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    slot.color_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    transition(list, backbuffer, D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(list, imported_nv12,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    (void)args;
    return true;
}

bool submit_fg_slot(Runtime& runtime, FgFrameSlot& slot, UINT64& fence_value) {
    if (FAILED(slot.list->Close())) return false;
    ID3D12CommandList* lists[] = {slot.list.Get()};
    runtime.queue->ExecuteCommandLists(1, lists);
    if(slot.provider_completion.value) {
        if(!slot.provider_completion.fence || FAILED(runtime.queue->Signal(slot.provider_completion.fence.Get(),slot.provider_completion.value)))return false;
        slot.provider_completion={};
    }
    return runtime.signal_queue(&fence_value);
}

struct EncodeJob {
    UINT copy_slot = 0;
    UINT frame_index = 0;
    UINT64 copy_fence = 0;
    ComPtr<ID3D12Resource> source_keepalive;
};

class AsyncEncoder {
public:
    NativeGpuPost* terminal_post = nullptr;
    std::string diagnostic_dir;
    AsyncEncoder(Runtime& runtime, ChainEncoder& encoder,
                 std::vector<EncodeCopySlot>& copy_slots, AsyncStats& stats)
        : runtime_(runtime), encoder_(encoder), copy_slots_(copy_slots), stats_(stats) {
        for (UINT index = 0; index < copy_slots_.size(); ++index)
            free_slots_.push_back(index);
    }

    ~AsyncEncoder() { finish(); }

    void start() { worker_ = std::thread(&AsyncEncoder::worker_loop, this); }

    bool enqueue(ComPtr<ID3D12Resource> source, D3D12_RESOURCE_STATES source_state,
                 FgFrameSlot* owner, UINT output_index,
                 D3D12_RESOURCE_STATES* owner_state = nullptr) {
        const auto wait_start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        bool blocked_on_slot = false;
        while (free_slots_.empty() && !failed_) {
            blocked_on_slot = true;
            if (condition_.wait_for(lock, std::chrono::seconds(30)) ==
                std::cv_status::timeout) {
                fail_locked("encode_slot_wait_timeout");
                break;
            }
        }
        const double waited = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wait_start).count();
        if (blocked_on_slot) {
            stats_.slot_wait_s += waited;
            ++stats_.slot_wait_count;
        }
        if (failed_ || free_slots_.empty() || !source) return false;
        const UINT copy_slot_index = free_slots_.front();
        free_slots_.pop_front();
        lock.unlock();

        if (!encoder_.begin_frame(copy_slot_index)) {
            lock.lock();
            free_slots_.push_front(copy_slot_index);
            fail_locked(encoder_.error);
            lock.unlock();
            condition_.notify_all();
            return false;
        }

        EncodeCopySlot& copy_slot = copy_slots_[copy_slot_index];
        const auto copy_start = std::chrono::steady_clock::now();
        if (FAILED(copy_slot.allocator->Reset()) ||
            FAILED(copy_slot.list->Reset(copy_slot.allocator.Get(), nullptr))) {
            lock.lock();
            free_slots_.push_front(copy_slot_index);
            fail_locked("encode_copy_list_reset");
            lock.unlock();
            condition_.notify_all();
            return false;
        }
        if (terminal_post) {
            terminal_post->record(runtime_.device.Get(),runtime_.motion_root.Get(),copy_slot.list.Get(),copy_slot_index,
                source.Get(),source_state,encoder_.opened(copy_slot_index));
            ++stats_.terminal_post_dispatch_count;
        } else {
        transition(copy_slot.list.Get(), source.Get(), source_state,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(copy_slot.list.Get(), encoder_.opened(copy_slot_index),
                   D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        copy_slot.list->CopyResource(encoder_.opened(copy_slot_index), source.Get());
        transition(copy_slot.list.Get(), encoder_.opened(copy_slot_index),
                   D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        transition(copy_slot.list.Get(), source.Get(),
                   D3D12_RESOURCE_STATE_COPY_SOURCE, source_state);
        }
        if (FAILED(copy_slot.list->Close())) {
            lock.lock();
            free_slots_.push_front(copy_slot_index);
            fail_locked("encode_copy_list_close");
            lock.unlock();
            condition_.notify_all();
            return false;
        }
        ID3D12CommandList* lists[] = {copy_slot.list.Get()};
        runtime_.queue->ExecuteCommandLists(1, lists);
        UINT64 copy_fence = 0;
        if (!runtime_.signal_queue(&copy_fence)) {
            lock.lock();
            free_slots_.push_front(copy_slot_index);
            fail_locked("encode_copy_signal");
            lock.unlock();
            condition_.notify_all();
            return false;
        }
        if (owner) {
            if (owner_state) *owner_state = source_state;
            else owner->color_state = source_state;
            owner->reuse_fence = copy_fence;
        }
        if(!diagnostic_dir.empty()) {
            std::string diagnostic_error;
            if(!runtime_.wait_fence(copy_fence,30000) ||
               !native_dump_resource(runtime_,source.Get(),source_state,diagnostic_dir+"/output"+std::to_string(output_index)+"-prepost.bin",stats_.diagnostic_post_readback_bytes,diagnostic_error) ||
               !native_dump_resource(runtime_,encoder_.opened(copy_slot_index),D3D12_RESOURCE_STATE_COMMON,diagnostic_dir+"/output"+std::to_string(output_index)+"-post.bin",stats_.diagnostic_post_readback_bytes,diagnostic_error)) {
                lock.lock();fail_locked("diagnostic_post:"+diagnostic_error);lock.unlock();return false;
            }
        }
        stats_.copy_submit_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - copy_start).count();
        {
            std::lock_guard<std::mutex> guard(mutex_);
            jobs_.push_back({copy_slot_index, output_index, copy_fence,
                             std::move(source)});
            stats_.encoder_queue_peak = std::max<UINT>(
                stats_.encoder_queue_peak, static_cast<UINT>(jobs_.size()));
            const UINT in_flight = static_cast<UINT>(jobs_.size() +
                (worker_busy_ ? 1u : 0u));
            stats_.max_in_flight = std::max(stats_.max_in_flight, in_flight);
        }
        condition_.notify_all();
        return true;
    }

    void finish() {
        if (!worker_.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            producer_done_ = true;
        }
        condition_.notify_all();
        worker_.join();
    }

    bool failed() const { return failed_.load(); }

    std::string error() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_;
    }

    size_t pending_jobs() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return jobs_.size();
    }

private:
    void fail_locked(const std::string& message) {
        failed_.store(true);
        if (error_.empty()) error_ = message.empty() ? "encode_worker" : message;
    }

    void worker_loop() {
        bool injected_failure = false;
        for (;;) {
            EncodeJob job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [&] {
                    return producer_done_ || !jobs_.empty();
                });
                if (jobs_.empty() && producer_done_) break;
                if (jobs_.empty()) continue;
                job = std::move(jobs_.front());
                jobs_.pop_front();
                worker_busy_ = true;
            }
            if (failed_.load()) {
                std::lock_guard<std::mutex> lock(mutex_);
                free_slots_.push_back(job.copy_slot);
                worker_busy_ = false;
                condition_.notify_all();
                continue;
            }
            if (std::getenv("XESS_FG_TEST_QSV_WORKER_FAILURE") &&
                !injected_failure) {
                injected_failure = true;
                std::lock_guard<std::mutex> lock(mutex_);
                fail_locked("injected_qsv_worker_failure");
                free_slots_.push_back(job.copy_slot);
                worker_busy_ = false;
                condition_.notify_all();
                continue;
            }
            double wait_s = 0.0;
            if (!wait_queue_fence(runtime_.fence.Get(), job.copy_fence, 30000,
                                  &wait_s)) {
                std::lock_guard<std::mutex> lock(mutex_);
                stats_.encode_worker_wait_s += wait_s;
                fail_locked("encode_copy_fence_timeout");
                free_slots_.push_back(job.copy_slot);
                worker_busy_ = false;
                condition_.notify_all();
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stats_.encode_worker_wait_s += wait_s;
            }
            const bool timestamp_ok=!terminal_post || terminal_post->collect(job.copy_slot);
            if(const char* delay=std::getenv("XESS_FG_TEST_ENCODER_DELAY_MS"))
                Sleep(static_cast<DWORD>(std::clamp(std::atoi(delay),0,1000)));
            const bool encoded = timestamp_ok && encoder_.encode_slot(job.copy_slot, job.frame_index);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!encoded) fail_locked(encoder_.error);
                free_slots_.push_back(job.copy_slot);
                worker_busy_ = false;
            }
            condition_.notify_all();
        }
    }

    Runtime& runtime_;
    ChainEncoder& encoder_;
    std::vector<EncodeCopySlot>& copy_slots_;
    AsyncStats& stats_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<UINT> free_slots_;
    std::deque<EncodeJob> jobs_;
    std::thread worker_;
    bool producer_done_ = false;
    bool worker_busy_ = false;
    std::atomic<bool> failed_{false};
    std::string error_;
};

bool tag_and_present(Runtime& runtime, const Args& args, UINT present_id,
                     bool reset, ID3D12GraphicsCommandList* tag_list,
                     ID3D12Resource* velocity_resource,
                     ID3D12Resource* depth_resource, UINT depth_width,
                     UINT depth_height, bool wait_after_present,
                     xefg_swapchain_present_status_t& status,
                     UINT& last_presented, AsyncStats& stats) {
    if (xellSleep(runtime.xell, present_id) != XELL_RESULT_SUCCESS ||
        xellAddMarkerData(runtime.xell, present_id, XELL_SIMULATION_START) !=
            XELL_RESULT_SUCCESS ||
        xellAddMarkerData(runtime.xell, present_id, XELL_SIMULATION_END) !=
            XELL_RESULT_SUCCESS ||
        xellAddMarkerData(runtime.xell, present_id, XELL_RENDERSUBMIT_START) !=
            XELL_RESULT_SUCCESS)
        return false;

    xefg_swapchain_d3d12_resource_data_t velocity{};
    velocity.type = XEFG_SWAPCHAIN_RES_MOTION_VECTOR;
    velocity.validity = XEFG_SWAPCHAIN_RV_UNTIL_NEXT_PRESENT;
    velocity.resourceSize = {static_cast<uint32_t>(args.width),
                             static_cast<uint32_t>(args.height)};
    velocity.pResource = velocity_resource;
    velocity.incomingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (!fg_ok(xefgSwapChainD3D12TagFrameResource(
                   runtime.xefg, tag_list, present_id, &velocity),
               "TagFrameResource(MOTION_VECTOR)")) return false;

    xefg_swapchain_d3d12_resource_data_t depth{};
    depth.type = XEFG_SWAPCHAIN_RES_DEPTH;
    depth.validity = XEFG_SWAPCHAIN_RV_UNTIL_NEXT_PRESENT;
    depth.resourceSize = {static_cast<uint32_t>(depth_width),
                          static_cast<uint32_t>(depth_height)};
    depth.pResource = depth_resource;
    depth.incomingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (!fg_ok(xefgSwapChainD3D12TagFrameResource(
                   runtime.xefg, tag_list, present_id, &depth),
               "TagFrameResource(DEPTH)")) return false;

    xefg_swapchain_frame_constant_data_t constants{};
    for (int i = 0; i < 16; ++i) {
        constants.viewMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        constants.projectionMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    }
    constants.motionVectorScaleX = 1.0f;
    constants.motionVectorScaleY = 1.0f;
    constants.resetHistory = reset ? 1u : 0u;
    constants.frameRenderTime = static_cast<float>(1000.0 / args.fps);
    if (!fg_ok(xefgSwapChainTagFrameConstants(
                   runtime.xefg, present_id, &constants),
               "TagFrameConstants") ||
        !fg_ok(xefgSwapChainSetPresentId(runtime.xefg, present_id),
               "SetPresentId")) return false;
    if (xellAddMarkerData(runtime.xell, present_id, XELL_RENDERSUBMIT_END) !=
            XELL_RESULT_SUCCESS ||
        xellAddMarkerData(runtime.xell, present_id, XELL_PRESENT_START) !=
            XELL_RESULT_SUCCESS)
        return false;

    const UINT before = runtime.native_swapchain->GetCurrentBackBufferIndex();
    const uint64_t captures_before = runtime.present_capture
        ? runtime.present_capture->captured_presents() : 0;
    DXGI_SWAP_CHAIN_DESC1 desc{};
    runtime.swapchain->GetDesc1(&desc);
    const bool tearing = (desc.Flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) != 0;
    const HRESULT presented = runtime.swapchain->Present(
        tearing ? 0 : 1, tearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
    xellAddMarkerData(runtime.xell, present_id, XELL_PRESENT_END);
    if (FAILED(presented)) return false;
    if (wait_after_present) {
        const auto wait_start = std::chrono::steady_clock::now();
        if (!runtime.wait_gpu()) return false;
        stats.present_required_wait_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wait_start).count();
        ++stats.present_required_wait_count;
    }
    if (!fg_ok(xefgSwapChainGetLastPresentStatus(runtime.xefg, &status),
               "GetLastPresentStatus")) return false;
    const UINT after = runtime.native_swapchain->GetCurrentBackBufferIndex();
    const UINT count = static_cast<UINT>(runtime.native_backbuffers.size());
    if (!count) return false;
    last_presented = (after + count - 1) % count;
    const uint64_t captures_after = runtime.present_capture
        ? runtime.present_capture->captured_presents() : 0;
    if (runtime.present_capture && status.framesPresented > 0 &&
        (!runtime.present_capture->capture_healthy() ||
         !runtime.present_capture->captured_resource() ||
         captures_after == captures_before)) {
        std::fprintf(stderr, "[full-gpu] pre-Present capture failed\n");
        return false;
    }
    std::fprintf(stderr,
                 "[full-gpu] present=%u native=%u->%u generated=%u result=%d "
                 "captures=%llu\n",
                 present_id, before, after, status.framesPresented,
                 static_cast<int>(status.frameGenResult),
                 static_cast<unsigned long long>(captures_after - captures_before));
    return true;
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const char c : value) {
        if (c == '\\' || c == '"') escaped.push_back('\\');
        if (c == '\n') { escaped += "\\n"; continue; }
        if (c == '\r') { escaped += "\\r"; continue; }
        escaped.push_back(c);
    }
    return escaped;
}

void write_full_gpu_report(const Options& options, const FullGpuDecoder& decoder,
                           const ChainEncoder& encoder, const AsyncStats& stats,
                           const GpuMotionCounters& motion,
                           const NativeGpuDepth* gpu_depth,
                           const NativeGpuTimers& gpu_timers, const NativeGpuPost& post,
                           const NativeSrEffects* effects,
                           UINT output_width, UINT output_height,
                           const DisPipeline* dis,
                           UINT input_frames, UINT output_frames, UINT active_slots,
                           UINT pending_jobs, double wall_s, bool pass,
                           const std::string& block) {
    if (options.report.empty()) return;
    std::ofstream out(options.report, std::ios::binary | std::ios::trunc);
    if (!out) return;
    // A successful report proves the real XeSS output-to-XeFG bridge and the
    // single-producer motion packet.  Depth remains classified separately:
    // CPU cache upload is never relabelled as STRICT_FULL_GPU.
    const bool cpu_depth_upload = !options.depth_dir.empty();
    const std::uint64_t depth_upload_bytes = cpu_depth_upload
        ? static_cast<std::uint64_t>(input_frames) * decoder.width() * decoder.height() * sizeof(float)
        : 0;
    const bool remote_depth = gpu_depth && motion.depth_inference_count == input_frames && input_frames > 0;
    const char *classification = "EXPERIMENTAL_ONLY";
    out << "{\n  \"schema_version\": \"xess-fg/full-gpu@2\",\n"
        << "  \"classification\": \"" << classification << "\",\n"
        << "  \"gate_status\": \"" << (pass ? (remote_depth ? "PASS_GPU_EXECUTION" : "PASS_COLOR_MOTION_DEPTH_PENDING") : "BLOCKED") << "\",\n"
        << "  \"blocked_on\": \"" << (pass ? (remote_depth ? "quality_lifetime_and_live_acceptance_pending" : "strict_gpu_depth_input") : "sr_fg_execution") << "\",\n"
        << "  \"block\": \"" << json_escape(block) << "\",\n"
        << "  \"dimensions\": {\"input\": {\"width\": " << decoder.width()
        << ", \"height\": " << decoder.height() << "}, \"requested\": {\"width\": "
        << output_width << ", \"height\": " << output_height
        << "}, \"working_aligned\": {\"width\": " << output_width
        << ", \"height\": " << output_height << "}, \"final_output\": {\"width\": "
        << output_width << ", \"height\": " << output_height << "}},\n"
        << "  \"frames\": {\"input\": " << input_frames
        << ", \"output\": " << output_frames
        << ", \"expected_output\": "
        << (options.gpu_mode == "sr" ? input_frames : input_frames ? input_frames * 2 - 1 : 0) << "},\n"
        << "  \"wall_seconds\": " << wall_s << ",\n"
        << "  \"pipeline\": {\"slots\": " << stats.slots
        << ", \"motion_backend\": \"" << json_escape(options.motion_backend) << "\""
        << ", \"mode\": \"" << options.gpu_mode << "\", \"motion_scale_applied\": " << (options.motion_backend=="gpu-block"?options.motion_scale:1)
        << ", \"motion_repair_applied\": \"" << (options.motion_backend=="gpu-block"?options.motion_repair:"none_provider_semantics_preserved") << "\""
        << ", \"max_in_flight\": " << stats.max_in_flight
        << ", \"decoder_queue_peak\": " << stats.decoder_queue_peak
        << ", \"encoder_queue_peak\": " << stats.encoder_queue_peak
        << ", \"active_slots\": " << active_slots
        << ", \"pending_jobs\": " << pending_jobs << "},\n"
        << "  \"waits\": {\"slot_wait_s\": " << stats.slot_wait_s
        << ", \"slot_wait_count\": " << stats.slot_wait_count
        << ", \"present_required_wait_s\": " << stats.present_required_wait_s
        << ", \"present_required_wait_count\": "
        << stats.present_required_wait_count
        << ", \"capture_wait_s\": " << stats.capture_wait_s
        << ", \"capture_wait_count\": " << stats.capture_wait_count
        << ", \"encode_worker_wait_s\": " << stats.encode_worker_wait_s
        << ", \"dis_graph_wait_s\": " << stats.dis_graph_wait_s
        << ", \"dis_graph_wait_count\": " << stats.dis_graph_wait_count << "},\n"
        << "  \"motion_packet\": {\"flow_pairs_requested\": "
        << motion.flow_pairs_requested << ", \"flow_pairs_computed\": "
        << motion.flow_pairs_computed << ", \"consumed_by_sr\": "
        << motion.consumed_by_sr << ", \"consumed_by_fg\": "
        << motion.consumed_by_fg << ", \"computed_equals_n_minus_1\": "
        << ((motion.flow_pairs_computed == (input_frames ? input_frames - 1 : 0)) ? "true" : "false")
        << ", \"producer\": \"" << json_escape(options.motion_backend) << " -> per-slot SRV/fence\","
        << " \"packet\": {\"frame_id\": " << motion.last_frame_id
        << ", \"prev_pts_90k\": " << motion.last_prev_pts_90k
        << ", \"current_pts_90k\": " << motion.last_current_pts_90k
        << ", \"source_size\": {\"width\": " << decoder.width()
        << ", \"height\": " << decoder.height() << "},"
        << " \"mv_direction\": \"current_to_previous\","
        << " \"confidence_occlusion\": \"GPU R32F confidence\","
        << " \"scene_cut\": " << (motion.last_scene_cut?"true":"false") << ", \"cpu_ready\": false,"
        << " \"producer_fence\": " << motion.last_producer_fence << "}},\n"
        << "  \"shared_core_counters\": {\"source_pair_analysis_count\": " << motion.flow_pairs_computed
        << ", \"directional_dispatch_count\": " << motion.directional_dispatch_count
        << ", \"directional_dispatch_includes_refine_research\": " << (options.motion_backend=="gpu-block"?"true":"false")
        << ", \"provider_adapter_dispatch_count\": " << motion.provider_adapter_dispatch_count
        << ", \"sr_consume_count\": " << motion.consumed_by_sr
        << ", \"fg_consume_count\": " << motion.consumed_by_fg
        << ", \"motion_after_sr_count\": 0, \"first_frame_self_analysis_count\": " << motion.first_frame_self_analysis_count
        << ", \"depth_inference_count\": " << motion.depth_inference_count
        << ", \"depth_pack_count\": " << motion.depth_pack_count
        << ", \"statistics_readback_bytes\": " << (gpu_depth ? gpu_depth->statistics_readback_bytes : 0) << "},\n"
        << "  \"gpu_dis\": {\"enabled\": "
        << (options.motion_backend=="gpu-dis" ? "true" : "false")
        << ", \"product_gpu_copies\": " << (dis ? dis->product_gpu_copies : 0)
        << ", \"mask_tagged\": false},\n"
        << "  \"depth_bridge\": {\"mode\": \""
        << (remote_depth ? (gpu_depth->rgb_input?"ONLINE_OPENVINO_GPU_RGB_NCHW_DX_BUFFER":"ONLINE_OPENVINO_GPU_REMOTE_NV12_AND_FP32_BUFFER") : cpu_depth_upload ? "CPU_CACHE_GPU_UPLOAD" : "constant_gpu_depth")
        << "\", \"cpu_upload_bytes\": " << depth_upload_bytes
        << ", \"gpu_copy_count\": " << (remote_depth ? motion.depth_gpu_copy_count : cpu_depth_upload ? input_frames : 0)
        << ", \"strict_interop_verified\": " << (remote_depth && pass ? "true" : "false")
        << ", \"cadence\": 1, \"inverse_depth_larger_is_nearer\": true"
        << ", \"normalize\": \"GPU finite minmax over 518x518; cubic a=-0.5; degenerate=0.5\""
        << ", \"temporal\": \"source-pixel current-to-previous warp; .25*confidence*(1-response)*(1-10*local-range)\""
        << ", \"inference_call_wall_seconds\": " << motion.depth_inference_seconds << "},\n"
        << "  \"depth_input_color\": {\"requested\": \"" << options.gpu_depth_input << "\", \"applied\": \""
        << (gpu_depth&&gpu_depth->rgb_input?"FFmpeg71_integer_BT601_BT709_limited_RGB8_cubic_u8_NCHW_ImageNet":"legacy_OpenVINO_NV12_fixed_BT601")
        << "\", \"matches_bt709_source\": " << (gpu_depth&&gpu_depth->rgb_input?"true":"false")
        << ", \"hdr\": \"UNSUPPORTED\", \"full_range\": \"UNSUPPORTED\", \"pixel_cpu_bytes\": 0"
        << ", \"nv12_gpu_copy_count\": " << (gpu_depth?gpu_depth->rgb_preprocess.nv12_copy_count:0)
        << ", \"tensor_gpu_copy_count\": " << (gpu_depth?gpu_depth->rgb_preprocess.tensor_copy_count:0)
        << ", \"preprocess_dispatch_count\": " << (gpu_depth?gpu_depth->rgb_preprocess.dispatch_count:0)
        << ", \"interop_acquire_count\": " << (gpu_depth?gpu_depth->model.rgb_acquire_count:0)
        << ", \"interop_release_event_wait_count\": " << (gpu_depth?gpu_depth->model.rgb_release_wait_count:0)
        << ", \"interop_release_event_wait_seconds\": " << (gpu_depth?gpu_depth->model.rgb_release_wait_seconds:0) << "},\n"
        << "  \"mask\": {\"computed\": true, \"sr_consumed\": " << (motion.consumed_by_sr > 0 ? "true" : "false")
        << ", \"depth_consumed\": " << (remote_depth ? "true" : "false")
        << ", \"mask_tagged\": false, \"definition\": \"bidirectional roundtrip, photometric residual, out-of-bounds"
        << (options.motion_backend=="gpu-block"?", block uniqueness":"; provider has no invented uniqueness") << "\"},\n"
        << "  \"provider_adapter\": {\"active\": " << (options.motion_backend=="gpu-block"?"false":"true")
        << ", \"flow\": \"R32G32 source flow preserved in packet; R16G16 SDK consumer adaptation\", \"depth_reliability\": \"GPU cycle consistency times photometric agreement, not constant confidence\"},\n"
        << "  \"sr_fg_bridge\": {\"verified\": " << (pass ? "true" : "false") << ","
        << " \"sr_source_persistent\": " << (pass ? "true" : "false") << ","
        << " \"source_resource\": \"per-slot sr_output XeSS D3D12 RGBA8\","
        << " \"proxy_copy\": \"sr_output COPY_SOURCE -> XeFG proxy COPY_DEST -> NON_PIXEL_SHADER_RESOURCE\","
        << " \"mask_tagged\": false},\n"
        << "  \"stages\": {\"decode_s\": " << stats.decode_s
        << ", \"record_submit_s\": " << stats.record_submit_s
        << ", \"present_s\": " << stats.present_s
        << ", \"copy_submit_s\": " << stats.copy_submit_s << "},\n"
        << "  \"drain\": {\"seconds\": " << stats.drain_s
        << ", \"result\": \"" << json_escape(stats.drain_result)
        << "\"},\n"
        << "  \"encode\": {\"imported\": " << encoder.imported_count
        << ", \"converted\": " << encoder.converted_count
        << ", \"encoded\": " << encoder.encoded_count
        << ", \"pts_mismatch\": " << encoder.pts_mismatch
        << ", \"pts_unknown\": " << encoder.pts_unknown << "},\n"
        << "  \"encode_settings\": {\"backend\": \"" << (encoder.software_mode()?"FFmpeg_terminal_RGB":"oneVPL_QSV")
        << "\", \"encoder\": \"" << encoder.terminal_encoder << "\", \"rgb_sha256\": \"" << encoder.software.rgb_sha256
        << "\", \"submission_counters_only\": " << (encoder.software_mode()?"true":"false")
        << ", \"rate_control\": \"" << (encoder.software_mode()?(encoder.terminal_encoder=="ffv1"?"lossless":"CRF18"):"ICQ20")
        << "\", \"target_usage\": 4, \"gop\": 48, \"b_frames\": 0, \"matrix\": \""
        << ColorMatrixName(encoder.output_matrix) << "\", \"range\": \"" << (encoder.terminal_encoder=="ffv1"?"full":"limited")
        << "\", \"onevpl_pipeline_used\": " << (encoder.software_mode()?"false":"true")
        << ", \"vpp_rgb_input_range\": \"full\", \"first_encoded_packet_from_encoder_start_ms\": "
        << (encoder.first_bit_seen?std::to_string(encoder.first_bit_ms-encoder.start_ms):"null") << "},\n"
        << "  \"boundaries\": {\n"
        << "    \"decoded_surface_cpu_map\": false,\n"
        << "    \"compressed_bitstream_buffer_bytes\": 4194304,\n"
        << "    \"full_frame_cpu_upload_bytes\": " << depth_upload_bytes << ",\n"
        << "    \"full_frame_cpu_readback_bytes\": " << motion.diagnostic_readback_bytes + stats.diagnostic_post_readback_bytes + encoder.terminal_readback_bytes << ",\n"
        << "    \"terminal_only_readback_bytes\": " << encoder.terminal_readback_bytes << ",\n"
        << "    \"terminal_rgb_pipe_bytes\": " << encoder.software.rgb_bytes << ",\n"
        << "    \"terminal_map_busy_retries\": " << encoder.terminal_map_busy_retries << ",\n"
        << "    \"diagnostic_full_frame_readback_enabled\": " << (!options.diagnostic_core_dir.empty()?"true":"false") << ",\n"
        << "    \"descriptor_neighbor_sentinel_enabled\": " << (std::getenv("XESS_GPU_TEST_DESCRIPTOR_SENTINEL")?"true":"false") << ",\n"
        << "    \"native_to_capture_copy_per_generated_frame\": 1,\n"
        << "    \"d3d12_to_d3d11_copy_per_output_frame\": " << (options.gpu_post=="on"?0:1) << ",\n"
        << "    \"d3d12_to_d3d11_post_compute_write_per_output_frame\": " << (options.gpu_post=="on"?1:0) << ",\n"
        << "    \"onevpl_vpp_import_copy_per_output_frame\": " << (encoder.software_mode()?0:1) << ",\n"
        << "    \"generated_capture_before_native_present\": true,\n"
        << "    \"presented_backbuffer_cpu_read_after_present\": false\n"
        << "  },\n"
        << "  \"capture\": {\"slots\": "
        << (options.slot_count ? options.slot_count : 4)
        << "},\n  \"depth_statistics\": [";
    if (gpu_depth) for (size_t i=0;i<gpu_depth->observed_statistics.size();++i) {
        if (i) out << ','; out << '[';
        for(size_t j=0;j<16;++j) {if(j) out << ','; out << gpu_depth->observed_statistics[i][j];}
        out << ']';
    }
    out << "],\n  \"depth_statistics_layout\": [\"raw_min\",\"raw_max\",\"raw_finite\",\"raw_nonfinite\",\"stable_min\",\"stable_max\",\"stable_mean\",\"stable_second_moment\",\"resized_min\",\"resized_max\",\"resized_mean\",\"resized_second_moment\",\"mask_min\",\"mask_max\",\"nonfinite\",\"mask_mean\"],\n"
        << "  \"scene\": {\"cut_count\": " << stats.scene_cut_count
        << ", \"metadata_bytes\": " << stats.scene_metadata_wait_count * 16
        << ", \"metadata_fence_wait_seconds\": " << stats.scene_metadata_wait_s
        << ", \"metadata_wait_count\": " << stats.scene_metadata_wait_count
        << ", \"external_reset_count\": " << stats.external_reset_count
        << ", \"cut_midpoint_policy\": \"duplicate_current_source_preserves_2Nminus1\"},\n"
        << "  \"terminal_post\": {\"requested\": \"" << options.gpu_post << "\", \"dispatch_count\": "
        << stats.terminal_post_dispatch_count << ", \"applied\": \"fixed_sharpen_0.25_terminal_once\"},\n"
        << "  \"sr_effects\": {\"five_frame\": " << (options.gpu_five_frame=="on"?"true":"false")
        << ", \"anti_stripe\": " << (options.gpu_anti_stripe=="on"?"true":"false")
        << ", \"placement\": \"after_sr_before_fg\", \"window\": \"current_plus_four_past_source_frames\""
        << ", \"frames\": " << (effects?effects->frames:0)
        << ", \"history_samples\": " << (effects?effects->history_samples:0)
        << ", \"history_resets\": " << (effects?effects->resets:0)
        << ", \"cpu_pixel_bytes\": 0, \"gpu_copy_per_source\": " << (effects?1:0) << "},\n"
        << "  \"ownership\": {\"external_input_accept_count\": " << motion.external_input_accept_count
        << ", \"mfx_pool_lease_release_count\": " << motion.pool_lease_release_count
        << ", \"previous_source_consumer_fence_count\": " << motion.previous_consumer_fence_count << "},\n"
        << "  \"gpu_timestamps\": {\"frames\": " << gpu_timers.frames << ", \"motion_seconds\": " << gpu_timers.motion_seconds
        << ", \"mask_seconds\": " << gpu_timers.mask_seconds << ", \"depth_post_seconds\": " << gpu_timers.depth_post_seconds
        << ", \"sr_seconds\": " << gpu_timers.sr_seconds << ", \"terminal_post_seconds\": " << post.gpu_seconds
        << ", \"openvino_gpu_model_seconds\": null, \"xefg_gpu_seconds\": null, \"parallel_stage_sums_are_not_wall_time\": true},\n"
        << "  \"stability\": {\"requested_seconds\": " << options.stability_seconds
        << ", \"compressed_input_loop\": " << (options.loop_input?"true":"false") << ", \"telemetry_retained_max_frames\": 4096},\n"
        << "  \"external_texture_ingress\": {\"synthetic_replay\": " << (options.synthetic_external?"true":"false")
        << ", \"separate_producer_queue\": " << (options.synthetic_external?"true":"false")
        << ", \"nv12_gpu_copy_count\": " << (options.synthetic_external?input_frames:0)
        << ", \"synthetic_missing_tick_at\": " << options.synthetic_gap_at
        << ", \"wgc_obs\": \"NOT_TESTED\"},\n"
        << "  \"live_wgc_obs\": \"NOT_TESTED\", \"original_pts\": \"elementary stream unavailable; synthesized CFR explicitly\", \"audio\": false\n}\n";
}

} // namespace

int xess_native_gpu_main(int argc, char** argv) {
    Options options;
    options.codec = "h264";
    options.max_frames = 8;
    options.slot_count = 4;
    options.motion_repair = "refine";
    options.motion_center_bias = 0.002f;
    if (!ParseArgs(argc, argv, options) || options.qsv_out.empty() ||
        options.shader_dir.empty() || options.fps <= 0.0) {
        std::fprintf(stderr,
            "Usage: xess-fg-full-gpu --input INPUT.h264 --codec h264 "
            "--max-frames N --fps INPUT_FPS --shader-dir DIR "
            "--qsv-out OUTPUT.h264 [--report report.json] "
            "[--slots 4] [--motion-backend gpu-block|gpu-dis|amd-of|zero-mv] "
            "[--depth-dir DIR] "
            "[--motion-repair refine|propagate|off]\n");
        return 2;
    }
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    if(options.stability_seconds>0) options.max_frames=1000000;
    if(options.synthetic_gap_at>=0 && !options.synthetic_external) {
        std::fprintf(stderr,"[full-gpu] synthetic gap needs synthetic external producer\n");return 2;
    }
    if(options.motion_backend!="gpu-block" && !xess_gpu::native_motion_factory) {
        std::fprintf(stderr,"[full-gpu] this shared core currently accepts only the GPU Block provider; other provider adapters are not connected\n"); return 4;
    }
    if(!options.depth_dir.empty()) {
        std::fprintf(stderr,"[full-gpu] precomputed CPU depth is not accepted by the shared native core; use --depth-model\n");return 4;
    }
    if(options.depth_model.empty()) {
        std::fprintf(stderr,"[full-gpu] online GPU depth requires --depth-model; no constant-depth fallback\n");return 4;
    }
    FullGpuDecoder decoder;
    if (!decoder.initialize(options)) {
        std::fprintf(stderr, "[full-gpu] decoder init failed: %s\n",
                     decoder.error.c_str());
        return 3;
    }
    const ColorSelection color = SelectColor(
        options, decoder.signal_info, decoder.width(), decoder.height());
    const UINT input_width = decoder.width();
    const UINT input_height = decoder.height();
    const UINT output_width = static_cast<UINT>(
        options.output_width > 0 ? options.output_width : input_width * (options.gpu_mode == "fg" ? 1u : 2u));
    const UINT output_height = static_cast<UINT>(
        options.output_height > 0 ? options.output_height : input_height * (options.gpu_mode == "fg" ? 1u : 2u));
    if (options.gpu_mode == "fg" && (output_width != input_width || output_height != input_height)) {
        std::fprintf(stderr, "[full-gpu] FG-only requires source dimensions\n"); return 4;
    }
    if (!output_width || !output_height) {
        std::fprintf(stderr, "[full-gpu] invalid SR output dimensions\n");
        return 4;
    }
    const bool sr_effects_on=options.gpu_five_frame=="on" || options.gpu_anti_stripe=="on";
    if(sr_effects_on && (options.gpu_mode=="fg" || (options.motion_backend!="gpu-block" && options.motion_backend!="gpu-dis" && options.motion_backend!="amd-of"))) {
        std::fprintf(stderr,"[full-gpu] SR effects require GPU Block/GPU DIS/AMD OF and SR or SR-FG\n");return 4;
    }
    const UINT slots = static_cast<UINT>(
        std::max(options.gpu_five_frame=="on"?5:1, std::min(8, options.slot_count)));
    if (slots < 2 && options.max_frames > 1) {
        std::fprintf(stderr,"[full-gpu] at least two slots required to retain previous source history\n"); return 4;
    }

    Args fg{};
    // XeFG's proxy and QSV endpoint operate on the real XeSS output.  Decode
    // and Block Motion resources remain input-sized in each FgFrameSlot.
    fg.width = output_width;
    fg.height = output_height;
    fg.frame_count = options.max_frames;
    fg.fps = options.fps;
    fg.device = options.adapter;
    fg.direct_capture = true;
    // Keep the common runtime's descriptor/depth setup identical for both
    // backends; GPU DIS records its own motion/mask graph below.
    fg.gpu_block_motion = true;
    fg.full_gpu = true;
    fg.shader_dir = options.shader_dir;
    fg.motion_repair = options.motion_repair;
    fg.motion_center_bias = options.motion_center_bias;
    fg.zero_mv_motion = options.motion_backend == "zero-mv";
    fg.depth_dir = !options.depth_model.empty() ? options.depth_model.c_str() : options.depth_dir.empty() ? nullptr : options.depth_dir.c_str();
    fg.motion_descriptor_slots = slots * 3;
    Runtime runtime;
    if (!init_runtime(fg, runtime)) {
        std::fprintf(stderr, "[full-gpu] XeFG runtime init failed\n");
        return 4;
    }
    DXGI_ADAPTER_DESC1 adapter_desc{};
    runtime.adapter->GetDesc1(&adapter_desc);
    ChainEncoder encoder;
    encoder.output_matrix=color.matrix;
    encoder.terminal_encoder=options.terminal_encoder;
    if (!encoder.init(runtime.device.Get(), adapter_desc.AdapterLuid,
                      output_width, output_height, slots,
                      options.qsv_out.c_str(), MFX_CODEC_AVC, 48,
                      options.fps * (options.gpu_mode == "sr" ? 1.0 : 2.0))) {
        std::fprintf(stderr, "[full-gpu] encoder init failed: %s\n",
                     encoder.error.c_str());
        return 5;
    }
    GpuSrRuntime sr;
    std::unique_ptr<xess_gpu::MotionProvider> native_provider;
    if(options.motion_backend!="gpu-block") {
        native_provider.reset(xess_gpu::native_motion_factory(options.motion_backend.c_str()));
        std::string provider_error;
        if(!native_provider || !native_provider->initialize(runtime.device.Get(),{{input_width,input_height},slots,true},provider_error)) {
            std::fprintf(stderr,"[full-gpu] provider unavailable: %s\n",provider_error.c_str());return 5;
        }
        sr.motion_provider=native_provider.get();
    }
    NativeGpuTimers gpu_timers;
    if(!gpu_timers.init(runtime.device.Get(),runtime.queue.Get(),slots)) {
        std::fprintf(stderr,"[full-gpu] GPU timestamp resources unavailable\n");return 5;
    }
    sr.gpu_timers=&gpu_timers;
    if (options.gpu_mode != "fg" && !sr.init(runtime.device.Get(), input_width, input_height,
                 output_width, output_height, options.xess_quality)) {
        std::fprintf(stderr, "[full-gpu] XeSS SR init failed\n");
        return 5;
    }
    std::string init_error;
    NativeSrEffects sr_effects;
    if(sr_effects_on) {
        if(!sr_effects.init(runtime.device.Get(),options.shader_dir,slots,output_width,output_height,
            options.gpu_anti_stripe=="on",options.gpu_five_frame=="on",init_error)) {
            std::fprintf(stderr,"[full-gpu] %s\n",init_error.c_str());return 5;
        }
        sr.effects=&sr_effects;
    }
    if(const char* sentinel=std::getenv("XESS_GPU_TEST_DESCRIPTOR_SENTINEL")) {
        sr.descriptor_sentinel_value=static_cast<float>(std::atof(sentinel));
        if(!std::isfinite(sr.descriptor_sentinel_value) || !chain_texture(runtime.device.Get(),1,1,DXGI_FORMAT_R8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,sr.mask_descriptor_sentinel,init_error))return 5;
    }
    if (!chain_pipeline(runtime.device.Get(), options.shader_dir, "native_provider_adapter", runtime.motion_root.Get(), sr.provider_adapter_pso, init_error) ||
        !chain_pipeline(runtime.device.Get(), options.shader_dir, "surface_gpu_mask", runtime.motion_root.Get(), sr.mask_pso, init_error) ||
        !chain_pipeline(runtime.device.Get(), options.shader_dir, "surface_gpu_lite_downsample", runtime.motion_root.Get(), sr.lite_downsample_pso, init_error) ||
        !chain_pipeline(runtime.device.Get(), options.shader_dir, "surface_gpu_lite_upsample", runtime.motion_root.Get(), sr.lite_upsample_pso, init_error)) {
        std::fprintf(stderr,"[full-gpu] %s\n",init_error.c_str()); return 5;
    }
    NativeGpuDepth gpu_depth;
    if (!options.depth_model.empty() && !gpu_depth.init(decoder.d3d11.Get(), runtime.device.Get(), input_width, input_height,
        output_width, output_height, slots, options.depth_model, options.shader_dir, init_error,
        decoder.decode_params.mfx.FrameInfo.Width,decoder.decode_params.mfx.FrameInfo.Height,options.gpu_depth_input=="ffmpeg-rgb")) {
        std::fprintf(stderr,"[full-gpu] %s\n",init_error.c_str()); return 5;
    }
    NativeGpuScene scene;
    if(!scene.init(runtime.device.Get(),runtime.motion_root.Get(),options.shader_dir,slots,init_error)) {
        std::fprintf(stderr,"[full-gpu] %s\n",init_error.c_str()); return 5;
    }
    NativeGpuPost terminal_post;
    if(options.gpu_post == "on" && !terminal_post.init(runtime.device.Get(),runtime.queue.Get(),runtime.motion_root.Get(),options.shader_dir,slots,init_error)) {
        std::fprintf(stderr,"[full-gpu] %s\n",init_error.c_str()); return 5;
    }
    NativeTextureReplay replay;
    if(options.synthetic_external && !replay.init(decoder.d3d11.Get(),runtime.device.Get(),
        decoder.decode_params.mfx.FrameInfo.Width,decoder.decode_params.mfx.FrameInfo.Height,slots,init_error)) {
        std::fprintf(stderr,"[full-gpu] %s\n",init_error.c_str());return 5;
    }
    if (std::getenv("XESS_FG_TEST_CLOSE_OUTPUT"))
        encoder.close_output_for_test();
    std::vector<FgFrameSlot> frame_slots;
    if (!init_fg_slots(runtime, input_width, input_height,
                       output_width, output_height, slots, frame_slots, options.motion_scale)) {
        std::fprintf(stderr, "[full-gpu] frame slot init failed\n");
        return 6;
    }
    DisPipeline dis;
    sr.effect_slots=&frame_slots;
    const bool use_legacy_dis=false; // Disconnected legacy graph is never a fallback.
    UINT64 dis_reuse_fence = 0;
    if (use_legacy_dis) {
        const std::string dis_shader_dir = options.shader_dir + "\\gpu-dis";
        if (!dis.init(runtime.device.Get(), dis_shader_dir,
                      input_width, input_height, input_width, input_height)) {
            std::fprintf(stderr, "[full-gpu] GPU DIS init failed: %s\n",
                         dis.err.c_str());
            return 8;
        }
        std::fprintf(stderr,
                     "[full-gpu] motion-backend=gpu-dis graph=shared-serial-island "
                     "mask=computed-not-xefg-tagged shader-dir=%s\n",
                     dis_shader_dir.c_str());
    }
    std::vector<EncodeCopySlot> copy_slots(slots);
    for (EncodeCopySlot& copy : copy_slots) {
        if (FAILED(runtime.device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&copy.allocator))) ||
            FAILED(runtime.device->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_DIRECT, copy.allocator.Get(), nullptr,
                IID_PPV_ARGS(&copy.list))) || FAILED(copy.list->Close())) {
            std::fprintf(stderr, "[full-gpu] encode copy slot init failed\n");
            return 7;
        }
    }

    AsyncStats stats;
    stats.slots = slots;
    AsyncEncoder async_encoder(runtime, encoder, copy_slots, stats);
    if(options.gpu_post == "on") async_encoder.terminal_post=&terminal_post;
    async_encoder.diagnostic_dir=options.diagnostic_core_dir;
    async_encoder.start();

    struct DecodePacket {
        UINT frame_index = 0;
        ComPtr<ID3D12Resource> imported;
        DecodeSurfaceLease decoded_surface;
    };
    std::mutex decode_mutex;
    std::condition_variable decode_condition;
    std::deque<DecodePacket> decode_queue;
    bool decoder_done = false;
    bool decoder_failed = false;
    std::string decoder_error;
    std::atomic<bool> decoder_stop{false};
    const auto decode_started = std::chrono::steady_clock::now();
    std::thread decoder_thread([&] {
        for (UINT index = 0; index < static_cast<UINT>(options.max_frames); ++index) {
            if (decoder_stop.load()) break;
            if (std::getenv("XESS_FG_TEST_EARLY_EOF") &&
                index >= std::max<UINT>(1, static_cast<UINT>(options.max_frames) / 2)) {
                std::lock_guard<std::mutex> lock(decode_mutex);
                decoder_failed = true;
                decoder_error = "injected_early_eof";
                break;
            }
            ComPtr<ID3D12Resource> imported;
            DecodeSurfaceLease decoded_surface;
            const int decoded = decoder.next(runtime.device.Get(), imported,
                                             &decoded_surface);
            if (decoded != 1) {
                std::lock_guard<std::mutex> lock(decode_mutex);
                decoder_failed = decoded < 0 || index != static_cast<UINT>(options.max_frames);
                decoder_error = decoded < 0 ? decoder.error : "decoder_eos";
                break;
            }
            std::unique_lock<std::mutex> lock(decode_mutex);
            decode_condition.wait_for(lock, std::chrono::seconds(30), [&] {
                return decoder_stop.load() || decode_queue.size() < slots;
            });
            if (decoder_stop.load()) break;
            if (decode_queue.size() >= slots) {
                decoder_failed = true;
                decoder_error = "decoder_queue_timeout";
                break;
            }
            decode_queue.push_back({index, std::move(imported),
                                    std::move(decoded_surface)});
            stats.decoder_queue_peak = std::max<UINT>(
                stats.decoder_queue_peak, static_cast<UINT>(decode_queue.size()));
            lock.unlock();
            decode_condition.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(decode_mutex);
            decoder_done = true;
        }
        decode_condition.notify_all();
    });

    const auto started = std::chrono::steady_clock::now();
    UINT input_frames = 0;
    UINT output_frames = 0;
    std::vector<float> true_depth;
    if (!options.depth_dir.empty())
        true_depth.resize(static_cast<size_t>(input_width) * input_height);
    GpuMotionCounters motion_counters;
    bool failed = false;
    std::string block;
    while (input_frames < static_cast<UINT>(options.max_frames) &&
           (options.stability_seconds<=0 || std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<options.stability_seconds)) {
        if(!options.cancel_file.empty() && GetFileAttributesA(options.cancel_file.c_str())!=INVALID_FILE_ATTRIBUTES) {
            failed=true;block="cancel_requested";break;
        }
        if (async_encoder.failed()) {
            failed = true;
            block = "encode_worker:" + async_encoder.error();
            break;
        }
        DecodePacket packet;
        {
            std::unique_lock<std::mutex> lock(decode_mutex);
            decode_condition.wait_for(lock, std::chrono::seconds(30), [&] {
                return !decode_queue.empty() || decoder_done ||
                       async_encoder.failed();
            });
            if (decode_queue.empty()) {
                failed = true;
                block = decoder_failed ? decoder_error :
                    (async_encoder.failed() ? "encode_worker:" + async_encoder.error()
                                            : "decoder_queue_timeout");
                break;
            }
            packet = std::move(decode_queue.front());
            decode_queue.pop_front();
        }
        decode_condition.notify_all();
        FgFrameSlot& slot = frame_slots[input_frames % slots];
        if (slot.reuse_fence) {
            const auto wait_start = std::chrono::steady_clock::now();
            bool timed_out = false;
            if (!runtime.wait_fence(slot.reuse_fence, 30000, &timed_out)) {
                failed = true;
                block = timed_out ? "frame_slot_wait_timeout" : "frame_slot_wait_failed";
                break;
            }
            stats.slot_wait_s += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wait_start).count();
            ++stats.slot_wait_count;
            slot.reuse_fence = 0;
            if(slot.timing_ready && !gpu_timers.collect(input_frames%slots,options.gpu_mode!="fg")) {failed=true;block="timestamp_order";break;}
            slot.timing_ready=false;
            if(!slot.external_input.frame.releasable()) {failed=true;block="frame_pool_lease_still_in_use";break;}
            slot.external_input={};++motion_counters.pool_lease_release_count;
            if (slot.depth_ready && !gpu_depth.validate_statistics(input_frames % slots, init_error)) {
                failed=true;block=init_error;break;
            }
            slot.depth_ready=false;
            slot.imported_nv12.Reset();
            slot.decoded_surface.reset();
        }
        slot.imported_nv12 = packet.imported;
        slot.decoded_surface = std::move(packet.decoded_surface);
        auto& external=slot.external_input;
        auto& metadata=external.frame.metadata;
        metadata.source_frame_id=input_frames;
        metadata.previous_source_frame_id=input_frames ? input_frames-1 : xess_gpu::kNoFrame;
        metadata.adapter_luid=runtime.device->GetAdapterLuid();
        metadata.reset=input_frames ? xess_gpu::ResetReason::None : xess_gpu::ResetReason::FirstFrame;
        if(options.synthetic_gap_at>=0) {
            // The synthetic producer intentionally omits one clock tick/ID.
            // This is an input-contract fixture, not a claim of WGC capture.
            metadata.source_frame_id=input_frames+(input_frames>=static_cast<UINT>(options.synthetic_gap_at)?1:0);
            metadata.previous_source_frame_id=input_frames ? frame_slots[(input_frames+slots-1)%slots].external_input.frame.metadata.source_frame_id : xess_gpu::kNoFrame;
            if(input_frames==static_cast<UINT>(options.synthetic_gap_at))metadata.reset=xess_gpu::ResetReason::InputDrop;
        }
        metadata.matrix=color.matrix==ColorMatrix::Bt709 ? xess_gpu::Matrix::Bt709 : xess_gpu::Matrix::Bt601;
        metadata.range=color.range==ColorRange::Full ? xess_gpu::Range::Full : xess_gpu::Range::Limited;
        metadata.transfer=xess_gpu::Transfer::Bt709;
        metadata.original_pts=static_cast<int64_t>(slot.decoded_surface.surface->Data.TimeStamp);
        metadata.original_time_base={1,90000};
        metadata.pts_known=slot.decoded_surface.surface->Data.TimeStamp!=MFX_TIMESTAMP_UNKNOWN;
        metadata.pts_synthesized=!metadata.pts_known;
        external.frame.color.resource=packet.imported;
        external.frame.color.format=DXGI_FORMAT_NV12;
        const auto decoded_desc=packet.imported->GetDesc();
        external.frame.color.allocation={static_cast<UINT>(decoded_desc.Width),decoded_desc.Height};
        external.frame.color.valid={decoder.decode_params.mfx.FrameInfo.CropX,decoder.decode_params.mfx.FrameInfo.CropY,input_width,input_height};
        external.frame.producer_pool_owner.Attach(new MfxPoolOwner(slot.decoded_surface.surface));
        mfxHDL surface_native=nullptr;mfxResourceType native_type{};
        if(slot.decoded_surface.surface->FrameInterface->GetNativeHandle(slot.decoded_surface.surface,&surface_native,&native_type)!=MFX_ERR_NONE) {
            failed=true;block="native_input_handle";break;
        }
        external.nv12_d3d11=reinterpret_cast<ID3D11Texture2D*>(surface_native);
        if(options.synthetic_external && !replay.produce(input_frames%slots,external,init_error)) {
            failed=true;block=init_error;break;
        }
        FgFrameSlot* previous = input_frames
            ? &frame_slots[(input_frames + slots - 1) % slots] : nullptr;
        const UINT proxy_index = runtime.swapchain->GetCurrentBackBufferIndex();
        if (proxy_index >= runtime.backbuffers.size()) {
            failed = true;
            block = "backbuffer_index";
            break;
        }
        const auto record_start = std::chrono::steady_clock::now();
        const float* depth_ptr = nullptr;
        if (!options.depth_dir.empty()) {
            const std::string depth_path = options.depth_dir + "\\depth_" +
                                           (input_frames < 1000000
                                                ? (std::string(6 - std::to_string(input_frames).size(), '0') +
                                                   std::to_string(input_frames))
                                                : std::to_string(input_frames)) + ".bin";
            std::ifstream depth_file(depth_path, std::ios::binary);
            const std::streamsize bytes = static_cast<std::streamsize>(
                true_depth.size() * sizeof(float));
            if (!depth_file || !depth_file.read(
                    reinterpret_cast<char*>(true_depth.data()), bytes)) {
                failed = true;
                block = "depth_file_missing_or_incomplete:" + depth_path;
                break;
            }
            depth_ptr = true_depth.data();
        }
        if (use_legacy_dis && dis_reuse_fence) {
            const auto dis_wait_start = std::chrono::steady_clock::now();
            bool timed_out = false;
            if (!runtime.wait_fence(dis_reuse_fence, 30000, &timed_out)) {
                failed = true;
                block = timed_out ? "gpu_dis_reuse_wait_timeout"
                                  : "gpu_dis_reuse_wait_failed";
                break;
            }
            stats.dis_graph_wait_s += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - dis_wait_start).count();
            ++stats.dis_graph_wait_count;
            dis_reuse_fence = 0;
        }
        const bool recorded = use_legacy_dis
            ? record_decoded_gpu_dis_frame(
                runtime, fg, slot, dis, packet.imported.Get(),
                runtime.backbuffers[proxy_index].Get(),
                static_cast<int>(input_frames), color)
            : record_external_gpu_source(runtime,fg,sr,motion_counters,slot,previous,input_frames,
                input_width,input_height,output_width,output_height,color,
                options.depth_model.empty() ? nullptr : &gpu_depth,input_frames%slots,
                input_frames ? static_cast<int>((input_frames+slots-1)%slots) : -1,scene,init_error);
        if (!recorded || !submit_fg_slot(runtime, slot, slot.work_fence)) {
            failed = true;
            block = "decoded_frame_submit:"+init_error;
            break;
        }
        slot.motion_producer_fence = slot.work_fence;
        slot.motion_packet.current_source_frame_id=metadata.source_frame_id;
        slot.motion_packet.previous_source_frame_id=metadata.previous_source_frame_id;
        slot.motion_packet.current_metadata=metadata;
        slot.motion_packet.has_reverse=input_frames>0;
        slot.motion_packet.produced={runtime.fence,slot.work_fence};
        if(!sr.motion_provider) {
            slot.motion_packet.current_to_previous.resource=slot.forward_flow;
            slot.motion_packet.previous_to_current.resource=slot.backward_flow;
            slot.motion_packet.confidence.resource=slot.confidence;
            slot.motion_packet.confidence_definition=xess_gpu::ConfidenceDefinition::BlockMatchPhotometricAndUniqueness;
            slot.motion_packet.confidence_semantics="local winner-versus-neighbour SAD margin; dynamic response adds roundtrip and photometric residual";
        }
        external.frame.consumers.push_back({xess_gpu::Consumer::Motion,{runtime.fence,slot.work_fence}});
        if(!options.depth_model.empty()) external.frame.consumers.push_back({xess_gpu::Consumer::Depth,{runtime.fence,slot.work_fence}});
        if(previous) {
            previous->external_input.frame.consumers.push_back({xess_gpu::Consumer::Motion,{runtime.fence,slot.work_fence}});
            previous->reuse_fence=std::max(previous->reuse_fence,slot.work_fence);
            // Only the next accepted pair closes this frame's registration.
            // Before this point it must not be declared releasable.
            previous->external_input.frame.consumers_registered=true;
            ++motion_counters.previous_consumer_fence_count;
        }
        if(!xess_gpu::matches(external.frame,slot.motion_packet)) {failed=true;block="motion_packet_identity";break;}
        // SDK reset flags are CPU parameters. Wait the analysis fence, not
        // queue-idle; decoder and encoder workers remain bounded.
        if(!use_legacy_dis) {
            const auto scene_wait_started=std::chrono::steady_clock::now();
            bool timed_out=false;
            if(!runtime.wait_fence(slot.work_fence,30000,&timed_out)) {
                failed=true;block="scene_metadata_fence";break;
            }
            stats.scene_metadata_wait_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-scene_wait_started).count();
            ++stats.scene_metadata_wait_count;
            float scene_values[4]{};
            if(!scene.read(input_frames%slots,scene_values)) {failed=true;block="scene_metadata_nonfinite";break;}
            if(!options.diagnostic_core_dir.empty()) {
                const std::string prefix=options.diagnostic_core_dir+"/frame"+std::to_string(input_frames)+"-";
                auto dump=[&](ID3D12Resource* r,D3D12_RESOURCE_STATES s,const char* name) {
                    return native_dump_resource(runtime,r,s,prefix+name+".bin",motion_counters.diagnostic_readback_bytes,init_error);
                };
                bool good=dump(slot.color.Get(),slot.color_state,"color") &&
                    dump(slot.luma.Get(),slot.luma_state,"luma") &&
                    dump(slot.forward_flow.Get(),slot.forward_state,"motion") &&
                    dump(slot.backward_flow.Get(),slot.backward_state,"motion-reverse") &&
                    dump(slot.confidence.Get(),slot.confidence_state,"confidence") &&
                    dump(slot.mask.Get(),slot.mask_state,"mask");
                if(good && !options.depth_model.empty()) {
                    auto& ds=gpu_depth.slots[input_frames%slots];
                    good=dump(ds.raw.resource.Get(),ds.raw.state,"depth-raw") &&
                        dump(ds.normalized.resource.Get(),ds.normalized.state,"depth-normalized") &&
                        dump(ds.stable.resource.Get(),ds.stable.state,"depth-stable") &&
                        dump(ds.resized.resource.Get(),ds.resized.state,"depth-resized");
                }
                if(!good) {failed=true;block=init_error;break;}
            }
            slot.motion_scene_cut=input_frames>0 && NativeGpuScene::cut(scene_values);
            const bool external_reset=input_frames>0 && metadata.reset!=xess_gpu::ResetReason::None;
            if(slot.motion_scene_cut || external_reset) {
                if(slot.motion_scene_cut) metadata.reset=xess_gpu::ResetReason::SceneCut;
                slot.motion_packet.current_metadata=metadata;
                if(slot.motion_scene_cut)++stats.scene_cut_count;
                if(external_reset)++stats.external_reset_count;
                slot.motion_scene_cut=true; // Both SDKs share reset and safe midpoint policy.
                std::fprintf(stderr,"[scene] cut source=%u difference=%.4f histogram=%.4f reliable=%.4f\n",
                    input_frames,scene_values[0],scene_values[1],scene_values[2]);
                if(FAILED(slot.allocator->Reset()) || FAILED(slot.list->Reset(slot.allocator.Get(),nullptr))) {
                    failed=true;block="scene_reset_record";break;
                }
                ID3D12DescriptorHeap* heaps[]={runtime.motion_heap.Get()};slot.list->SetDescriptorHeaps(1,heaps);
                create_uav(runtime.device.Get(),runtime.motion_heap.Get(),slot.descriptor_base+60,runtime.motion_descriptor_stride,
                    slot.velocity.Get(),DXGI_FORMAT_R16G16_FLOAT);
                transition(slot.list.Get(),slot.velocity.Get(),slot.velocity_state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                const float zero[4]{};
                slot.list->ClearUnorderedAccessViewFloat(gpu_descriptor(runtime.motion_heap.Get(),slot.descriptor_base+60,runtime.motion_descriptor_stride),
                    cpu_descriptor(runtime.motion_heap.Get(),slot.descriptor_base+60,runtime.motion_descriptor_stride),slot.velocity.Get(),zero,0,nullptr);
                transition(slot.list.Get(),slot.velocity.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                slot.velocity_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                if(!options.depth_model.empty()) gpu_depth.record(input_frames%slots,-1,slot.list.Get(),slot.forward_flow.Get(),slot.confidence.Get(),slot.mask.Get(),true);
                if(!submit_fg_slot(runtime,slot,slot.work_fence) || !runtime.wait_fence(slot.work_fence,30000,&timed_out)) {
                    failed=true;block="scene_reset_submit";break;
                }
            }
            if(!record_gpu_consumers(runtime,sr,motion_counters,slot,runtime.backbuffers[proxy_index].Get(),input_width,input_height,
                input_frames==0 || slot.motion_scene_cut,options.gpu_mode!="fg") || !submit_fg_slot(runtime,slot,slot.work_fence)) {
                failed=true;block="consumer_submit";break;
            }
        }
        motion_counters.last_frame_id = slot.motion_frame_id;
        motion_counters.last_prev_pts_90k = slot.motion_prev_pts_90k;
        motion_counters.last_current_pts_90k = slot.motion_current_pts_90k;
        motion_counters.last_producer_fence = slot.motion_producer_fence;
        motion_counters.last_scene_cut = slot.motion_scene_cut;
        if (use_legacy_dis)
            dis_reuse_fence = slot.work_fence;
        stats.record_submit_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - record_start).count();
        const auto present_start = std::chrono::steady_clock::now();
        if (options.gpu_mode != "sr") {
        xefg_swapchain_present_status_t status{};
        UINT native_index = 0;
        ID3D12Resource* depth_resource = !options.depth_model.empty()
            ? gpu_depth.slots[input_frames % slots].resized.resource.Get()
            : options.depth_dir.empty() ? runtime.depth.Get() : slot.depth.Get();
        if (!tag_and_present(runtime, fg, input_frames + 1,
                             input_frames == 0 || slot.motion_scene_cut, slot.list.Get(),
                             slot.velocity.Get(), depth_resource,
                             options.depth_dir.empty() ? output_width : input_width,
                             options.depth_dir.empty() ? output_height : input_height,
                             false,
                             status, native_index, stats)) {
            failed = true;
            block = "present";
            break;
        }
        stats.present_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - present_start).count();
        slot.motion_consumed_by_fg = true;
        ++motion_counters.consumed_by_fg;
        const uint32_t expected_generated = input_frames == 0 || slot.motion_scene_cut ? 0u : 1u;
        if (status.framesPresented != expected_generated) {
            failed = true;
            block = "unexpected_generated_count=" +
                    std::to_string(status.framesPresented);
            break;
        }
        if (input_frames > 0) {
            ComPtr<ID3D12Resource> generated = slot.motion_scene_cut ? slot.sr_output : runtime.present_capture
                ? runtime.present_capture->captured_resource() : nullptr;
            if (!generated || !async_encoder.enqueue(
                    std::move(generated), slot.motion_scene_cut ? slot.sr_state : D3D12_RESOURCE_STATE_COMMON,
                    nullptr, output_frames)) {
                failed = true;
                block = "encode_generated:" + async_encoder.error();
                break;
            }
            ++output_frames;
        }
        }
        ComPtr<ID3D12Resource> source = slot.sr_output;
        if (!async_encoder.enqueue(std::move(source), slot.sr_state,
                                   &slot, output_frames, &slot.sr_state)) {
            failed = true;
            block = "encode_source:" + async_encoder.error();
            break;
        }
        ++output_frames;
        external.frame.consumers.push_back({xess_gpu::Consumer::Encode,{runtime.fence,slot.reuse_fence}});
        if(options.gpu_mode!="fg") external.frame.consumers.push_back({xess_gpu::Consumer::Sr,{runtime.fence,slot.reuse_fence}});
        if(options.gpu_mode!="sr") external.frame.consumers.push_back({xess_gpu::Consumer::Fg,{runtime.fence,slot.reuse_fence}});
        if(options.gpu_post=="on") external.frame.consumers.push_back({xess_gpu::Consumer::Post,{runtime.fence,slot.reuse_fence}});
        external.frame.consumers_registered=false;
        ++input_frames;
        if ((input_frames % 25) == 0)
            std::fprintf(stderr, "[full-gpu] processed %u/%d -> %u\n",
                         input_frames, options.max_frames, output_frames);
    }
    decoder_stop.store(true);
    decode_condition.notify_all();
    if (decoder_thread.joinable()) decoder_thread.join();
    stats.decode_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - decode_started).count();
    if (!failed && decoder_failed) {
        failed = true;
        block = decoder_error;
    }
    async_encoder.finish();
    if (async_encoder.failed()) {
        failed = true;
        block = "encode_worker:" + async_encoder.error();
    }

    const auto drain_start = std::chrono::steady_clock::now();
    bool drain_ok = true;
    bool drain_timeout = false;
    UINT active_slots = 0;
    for (FgFrameSlot& slot : frame_slots) {
        // Decoder/producer has stopped; there can be no additional next-pair
        // consumers, including for the final accepted source frame.
        slot.external_input.frame.consumers_registered=true;
        const UINT64 fence = slot.reuse_fence ? slot.reuse_fence : slot.work_fence;
        if (fence) {
            bool timed_out = false;
            if (!runtime.wait_fence(fence, 30000, &timed_out)) {
                drain_ok = false;
                drain_timeout = drain_timeout || timed_out;
            }
        }
        if (slot.depth_ready && drain_ok && !gpu_depth.validate_statistics(static_cast<UINT>(&slot-frame_slots.data()),init_error)) {
            drain_ok=false;block=init_error;
        }
        if(slot.timing_ready && drain_ok && !gpu_timers.collect(static_cast<UINT>(&slot-frame_slots.data()),options.gpu_mode!="fg")) {drain_ok=false;block="timestamp_order";}
        if(slot.external_input.frame.producer_pool_owner && drain_ok) {
            if(!slot.external_input.frame.releasable()) {drain_ok=false;block="drain_pool_lease_in_use";++active_slots;}
            else {slot.external_input={};++motion_counters.pool_lease_release_count;}
        }
        slot.reuse_fence = 0;
        slot.work_fence = 0;
        slot.imported_nv12.Reset();
        slot.decoded_surface.reset();
    }
    if (runtime.present_capture && !runtime.present_capture->capture_healthy()) {
        drain_ok = false;
        if (block.empty()) block = "capture_failed";
    }
    if (!failed && drain_ok && output_frames && !encoder.drain()) {
        drain_ok = false;
        if (block.empty()) block = "encoder_drain";
    }
    stats.drain_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - drain_start).count();
    stats.drain_result = drain_ok ? "ok" : (drain_timeout ? "timeout" : "failed");
    if (!drain_ok) failed = true;
    if (runtime.present_capture) {
        stats.capture_wait_count = static_cast<UINT>(
            runtime.present_capture->capture_wait_count());
        stats.capture_wait_s = runtime.present_capture->capture_wait_seconds();
    }
    const double wall_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const UINT expected = options.gpu_mode == "sr" ? input_frames : input_frames ? input_frames * 2 - 1 : 0;
    const UINT pending_jobs = static_cast<UINT>(async_encoder.pending_jobs());
    const bool pass = !failed && !decoder_failed && !async_encoder.failed() &&
        (options.stability_seconds>0 ? wall_s>=options.stability_seconds : input_frames == static_cast<UINT>(options.max_frames)) &&
        output_frames == expected && encoder.imported_count == output_frames &&
        encoder.converted_count == output_frames &&
        encoder.encoded_count == output_frames &&
        encoder.pts_mismatch == 0 && encoder.pts_unknown == 0 &&
        active_slots == 0 && pending_jobs == 0 && stats.drain_result == "ok";
    motion_counters.depth_inference_count = gpu_depth.inference_count;
    motion_counters.depth_gpu_copy_count = gpu_depth.gpu_copy_count;
    motion_counters.depth_pack_count = gpu_depth.pack_count;
    motion_counters.depth_inference_seconds = gpu_depth.inference_seconds;
    write_full_gpu_report(options, decoder, encoder, stats, motion_counters,
                          options.depth_model.empty() ? nullptr : &gpu_depth,
                          gpu_timers, terminal_post, sr.effects,
                          output_width, output_height,
                          use_legacy_dis ? &dis : nullptr,
                          input_frames,
                          output_frames, active_slots, pending_jobs, wall_s,
                          pass, block);
    std::fprintf(stderr,
                 "[full-gpu] input=%u output=%u expected=%u encoded=%u "
                 "wall=%.3fs pass=%u block=%s\n",
                 input_frames, output_frames, expected, encoder.encoded_count,
                 wall_s, pass ? 1u : 0u, block.c_str());
    return pass ? 0 : 20;
}
#ifndef XESS_NATIVE_GPU_CORE_LIBRARY
int main(int argc,char** argv) {return xess_native_gpu_main(argc,argv);}
#endif
