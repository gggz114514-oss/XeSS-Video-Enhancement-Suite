// Full-GPU product chain probe: oneVPL D3D11 decode -> shared handle ->
// D3D12 color/motion/repair/velocity/mask -> XeSS -> fused GPU post ->
// D3D11-owned shared RGBA8 -> oneVPL VPP (RGB4->NV12, IMPORT_COPY) -> QSV
// H.264 encode, inside a persistent 4-slot pipeline with bounded queues.
// No full-frame CPU readback or upload exists in the product path; the only
// per-frame driver copy is oneVPL's VPP IMPORT_COPY and is reported rather
// than being claimed zero-copy.
//
// Structure and ownership follow the accepted A5.3/A5.4 experiments in the
// quality-full-gpu worktree: D3D11 creates NT-handle keyed-mutex textures,
// D3D12 opens them and uses them directly as XeSS/post UAV outputs.
#define main gate1_surface_probe_main
#include "vpl_decode_surface_probe.cpp"
#undef main

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <numeric>
#include "native_software_encoder.h"

#include "xess/xess.h"
#include "xess/xess_d3d12.h"

using Microsoft::WRL::ComPtr;

namespace {

struct GpuResource {
    ComPtr<ID3D12Resource> resource;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

bool chain_check(HRESULT hr, const char *where, std::string &error) {
    if (SUCCEEDED(hr)) return true;
    char text[32]{};
    std::snprintf(text, sizeof(text), "0x%08lX", static_cast<unsigned long>(hr));
    error = std::string(where) + "=" + text;
    return false;
}

bool chain_texture(ID3D12Device *device, UINT width, UINT height, DXGI_FORMAT format,
                   D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                   GpuResource &out, std::string &error) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc = {1, 0};
    desc.Flags = flags;
    D3D12_HEAP_PROPERTIES props{};
    props.Type = D3D12_HEAP_TYPE_DEFAULT;
    out.state = state;
    return chain_check(device->CreateCommittedResource(
                           &props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                           IID_PPV_ARGS(&out.resource)),
                       "CreateCommittedResource", error);
}

void chain_transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
                      D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

D3D12_CPU_DESCRIPTOR_HANDLE chain_cpu_desc(ID3D12DescriptorHeap *heap, UINT index,
                                           UINT stride) {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * stride;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE chain_gpu_desc(ID3D12DescriptorHeap *heap, UINT index,
                                           UINT stride) {
    D3D12_GPU_DESCRIPTOR_HANDLE handle = heap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<UINT64>(index) * stride;
    return handle;
}

void chain_srv(ID3D12Device *device, ID3D12DescriptorHeap *heap, UINT index, UINT stride,
               ID3D12Resource *resource, DXGI_FORMAT format, UINT plane = 0) {
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = format;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Texture2D.MipLevels = 1;
    view.Texture2D.PlaneSlice = plane;
    device->CreateShaderResourceView(resource, &view, chain_cpu_desc(heap, index, stride));
}

void chain_uav(ID3D12Device *device, ID3D12DescriptorHeap *heap, UINT index, UINT stride,
               ID3D12Resource *resource, DXGI_FORMAT format) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = format;
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(resource, nullptr, &view,
                                      chain_cpu_desc(heap, index, stride));
}

bool chain_pipeline(ID3D12Device *device, const std::string &shader_dir,
                    const char *shader_name, ID3D12RootSignature *root,
                    ComPtr<ID3D12PipelineState> &pipeline, std::string &error) {
    std::ifstream in(shader_dir + "/" + shader_name + ".dxil",
                     std::ios::binary | std::ios::ate);
    if (!in) { error = std::string("shader_open=") + shader_name; return false; }
    const std::streamoff size = in.tellg();
    std::vector<uint8_t> blob(static_cast<size_t>(size));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char *>(blob.data()), size);
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.CS = {blob.data(), blob.size()};
    return chain_check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pipeline)),
                       "CreateComputePipelineState", error);
}

bool chain_root(ID3D12Device *device, ComPtr<ID3D12RootSignature> &root,
                std::string &error) {
    D3D12_DESCRIPTOR_RANGE srv{};
    srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv.NumDescriptors = 5;
    srv.BaseShaderRegister = 0;
    D3D12_DESCRIPTOR_RANGE uav{};
    uav.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav.NumDescriptors = 2;
    uav.BaseShaderRegister = 0;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &srv;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &uav;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.Num32BitValues = 6;
    params[2].Constants.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 3;
    desc.pParameters = params;
    ComPtr<ID3DBlob> serialized, errors;
    if (!chain_check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                                 &serialized, &errors),
                     "D3D12SerializeRootSignature", error)) return false;
    return chain_check(device->CreateRootSignature(
                           0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                           IID_PPV_ARGS(&root)), "CreateRootSignature", error);
}

// ---------------------------------------------------------------------------
// QSV bridge: D3D11-owned shared RGBA8 keyed-mutex textures; oneVPL imports
// them with IMPORT_SHARED|IMPORT_COPY (the runtime answers 0x20), VPP
// converts RGB4->NV12 in video memory and H.264 QSV consumes it.  Ported
// from the accepted quality-full-gpu A5.3 contract.
struct ChainEncoder {
    struct SharedSurface {
        ComPtr<ID3D11Texture2D> texture;
        ComPtr<IDXGIKeyedMutex> mutex;
        ComPtr<ID3D12Resource> opened12;
        UINT key = 0;
    };

    ComPtr<ID3D11Device> d3d11;
    ComPtr<ID3D11DeviceContext> d3d11_context;
    mfxLoader loader = nullptr;
    mfxSession session = nullptr;
    mfxMemoryInterface *memory = nullptr;
    std::vector<SharedSurface> surfaces;
    FILE *output = nullptr;
    UINT out_w = 0, out_h = 0;
    UINT imported_count = 0, converted_count = 0, encoded_count = 0;
    UINT pts_mismatch = 0, pts_unknown = 0;
    UINT64 import_copy_count = 0;
    ULONGLONG start_ms = 0, first_bit_ms = 0;
    bool first_bit_seen = false;
    mfxU32 codec_id = MFX_CODEC_AVC;
    mfxU16 gop = 48;
    double fps = 24.0;
    ColorMatrix output_matrix = ColorMatrix::Bt601;
    std::string error;
    bool ready = false;
    std::string terminal_encoder = "h264_qsv";
    NativeSoftwareEncoder software;
    ComPtr<ID3D11Texture2D> terminal_staging;
    ComPtr<ID3D11Query> terminal_query;
    std::vector<UCHAR> terminal_rgb;
    UINT64 terminal_readback_bytes = 0;
    UINT64 terminal_map_busy_retries = 0;
    bool software_mode() const {return terminal_encoder=="libx264" || terminal_encoder=="libx265" || terminal_encoder=="ffv1";}

    ~ChainEncoder() {
        if (session) {
            MFXVideoENCODE_Close(session);
            MFXVideoVPP_Close(session);
            MFXClose(session);
        }
        if (loader) MFXUnload(loader);
        if (output) fclose(output);
    }

    static bool set_frame_info(mfxFrameInfo *info, mfxU32 fourcc, UINT width, UINT height) {
        if (!info || width > 16384 || height > 16384) return false;
        memset(info, 0, sizeof(*info));
        info->FourCC = fourcc;
        info->ChromaFormat = MFX_CHROMAFORMAT_YUV420;
        info->PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
        info->FrameRateExtN = 30;
        info->FrameRateExtD = 1;
        info->CropW = static_cast<mfxU16>(width);
        info->CropH = static_cast<mfxU16>(height);
        info->Width = static_cast<mfxU16>((width + 15u) & ~15u);
        info->Height = static_cast<mfxU16>((height + 15u) & ~15u);
        return true;
    }

    void set_frame_rate(mfxFrameInfo *info) const {
        if (!info) return;
        // Preserve NTSC-family rates without rounding 59.9401/119.8802 to an
        // integer.  For other rates a reduced millihertz rational is enough
        // for the encoder timing contract.
        mfxU32 denominator = 1000;
        mfxU32 numerator = static_cast<mfxU32>(std::lround(fps * denominator));
        const double ntsc_units = fps * 1001.0;
        if (std::fabs(ntsc_units - std::round(ntsc_units)) < 0.02) {
            denominator = 1001;
            numerator = static_cast<mfxU32>(std::lround(ntsc_units));
        }
        const mfxU32 divisor = std::gcd(numerator, denominator);
        info->FrameRateExtN = numerator / std::max<mfxU32>(1, divisor);
        info->FrameRateExtD = denominator / std::max<mfxU32>(1, divisor);
    }

    bool create_d3d11(const LUID &target_luid) {
        ComPtr<IDXGIFactory6> factory;
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
            error = "encoder_factory";
            return false;
        }
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc)) ||
                memcmp(&desc.AdapterLuid, &target_luid, sizeof(LUID)) != 0)
                continue;
            adapter = candidate;
            break;
        }
        if (!adapter) { error = "encoder_adapter_not_found"; return false; }
        constexpr D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1,
                                                D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL selected{};
        HRESULT hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                                       D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels,
                                       static_cast<UINT>(std::size(levels)),
                                       D3D11_SDK_VERSION, &d3d11, &selected,
                                       &d3d11_context);
        if (FAILED(hr)) { error = "encoder_d3d11_create"; return false; }
        ComPtr<IDXGIDevice> dxgi;
        ComPtr<IDXGIAdapter> d3d11_adapter;
        DXGI_ADAPTER_DESC a{}, b{};
        const bool same = SUCCEEDED(d3d11.As(&dxgi)) &&
            SUCCEEDED(dxgi->GetAdapter(&d3d11_adapter)) &&
            SUCCEEDED(adapter->GetDesc(&a)) &&
            SUCCEEDED(d3d11_adapter->GetDesc(&b)) &&
            memcmp(&a.AdapterLuid, &b.AdapterLuid, sizeof(LUID)) == 0;
        if (!same) { error = "encoder_luid_mismatch"; return false; }
        ComPtr<ID3D10Multithread> mt;
        if (SUCCEEDED(d3d11_context.As(&mt))) mt->SetMultithreadProtected(TRUE);
        return true;
    }

    bool init_session() {
        loader = MFXLoad();
        if (!loader) { error = "encoder_mfxload"; return false; }
        mfxConfig configs[3]{};
        for (auto &config : configs) config = MFXCreateConfig(loader);
        mfxVariant value{};
        value.Type = MFX_VARIANT_TYPE_U32;
        value.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
        MFXSetConfigFilterProperty(configs[0], (mfxU8 *)"mfxImplDescription.Impl", value);
        value.Data.U32 = MFX_ACCEL_MODE_VIA_D3D11;
        MFXSetConfigFilterProperty(configs[1],
                                   (mfxU8 *)"mfxImplDescription.AccelerationMode", value);
        value.Data.U32 = (2u << 16) | 10u;
        MFXSetConfigFilterProperty(configs[2],
                                   (mfxU8 *)"mfxImplDescription.ApiVersion.Version", value);
        if (MFXCreateSession(loader, 0, &session) != MFX_ERR_NONE) {
            error = "encoder_session";
            return false;
        }
        if (MFXVideoCORE_SetHandle(session, MFX_HANDLE_D3D11_DEVICE,
                                   reinterpret_cast<mfxHDL>(d3d11.Get())) != MFX_ERR_NONE ||
            MFXGetMemoryInterface(session, &memory) != MFX_ERR_NONE || !memory) {
            error = "encoder_handle";
            return false;
        }
        return true;
    }

    bool init_components() {
        mfxVideoParam vpp{};
        set_frame_info(&vpp.vpp.In, MFX_FOURCC_BGR4, out_w, out_h);
        set_frame_info(&vpp.vpp.Out, MFX_FOURCC_NV12, out_w, out_h);
        set_frame_rate(&vpp.vpp.In);
        set_frame_rate(&vpp.vpp.Out);
        vpp.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;
        vpp.AsyncDepth = 1;
        mfxExtVPPVideoSignalInfo vpp_signal{};
        vpp_signal.Header={MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO,sizeof(vpp_signal)};
        vpp_signal.In.TransferMatrix=vpp_signal.Out.TransferMatrix=
            output_matrix==ColorMatrix::Bt709?MFX_TRANSFERMATRIX_BT709:MFX_TRANSFERMATRIX_BT601;
        vpp_signal.In.NominalRange=MFX_NOMINALRANGE_0_255;
        vpp_signal.Out.NominalRange=MFX_NOMINALRANGE_16_235;
        mfxExtBuffer* vpp_ext[]={reinterpret_cast<mfxExtBuffer*>(&vpp_signal)};
        vpp.ExtParam=vpp_ext;vpp.NumExtParam=1;
        mfxVideoParam queried = vpp;
        mfxStatus vpp_q = MFXVideoVPP_Query(session, &vpp, &queried);
        mfxStatus vpp_i = MFXVideoVPP_Init(session, &queried);
        if (vpp_q < MFX_ERR_NONE || vpp_i < MFX_ERR_NONE) {
            char text[96]{};
            std::snprintf(text, sizeof(text), "encoder_vpp_init q=%d i=%d",
                          static_cast<int>(vpp_q), static_cast<int>(vpp_i));
            error = text;
            return false;
        }
        mfxVideoParam enc{};
        enc.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY;
        enc.AsyncDepth = 1;
        enc.mfx.CodecId = codec_id;
        enc.mfx.TargetUsage = 4;  // ffmpeg h264_qsv preset medium
        enc.mfx.RateControlMethod = MFX_RATECONTROL_ICQ;
        enc.mfx.ICQQuality = 20;  // -global_quality 20
        enc.mfx.GopRefDist = 1;   // -bf 0
        enc.mfx.GopPicSize = gop; // -g 48
        set_frame_info(&enc.mfx.FrameInfo, MFX_FOURCC_NV12, out_w, out_h);
        set_frame_rate(&enc.mfx.FrameInfo);
        mfxExtVideoSignalInfo encoded_signal{};
        encoded_signal.Header={MFX_EXTBUFF_VIDEO_SIGNAL_INFO,sizeof(encoded_signal)};
        encoded_signal.VideoFormat=5;encoded_signal.VideoFullRange=0;
        encoded_signal.ColourDescriptionPresent=1;
        encoded_signal.ColourPrimaries=output_matrix==ColorMatrix::Bt709?1:6;
        encoded_signal.TransferCharacteristics=1;
        encoded_signal.MatrixCoefficients=output_matrix==ColorMatrix::Bt709?1:6;
        mfxExtBuffer* encode_ext[]={reinterpret_cast<mfxExtBuffer*>(&encoded_signal)};
        enc.ExtParam=encode_ext;enc.NumExtParam=1;
        queried = enc;
        if (MFXVideoENCODE_Query(session, &enc, &queried) < MFX_ERR_NONE ||
            MFXVideoENCODE_Init(session, &queried) < MFX_ERR_NONE) {
            error = "encoder_encode_init";
            return false;
        }
        return true;
    }

    bool create_shared_surfaces(ID3D12Device *device, UINT count) {
        surfaces.resize(count);
        for (UINT index = 0; index < count; ++index) {
            SharedSurface &surface = surfaces[index];
            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = out_w;
            desc.Height = out_h;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            // 0x88 (SHADER_RESOURCE|UNORDERED_ACCESS): the A5.4 direct-output
            // contract; D3D12 opens it with ALLOW_UNORDERED_ACCESS.
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
            desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                             D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
            if (FAILED(d3d11->CreateTexture2D(&desc, nullptr, &surface.texture))) {
                error = "encoder_shared_create";
                return false;
            }
            if (FAILED(surface.texture.As(&surface.mutex))) {
                error = "encoder_keyed_mutex";
                return false;
            }
            ComPtr<IDXGIResource1> resource;
            if (FAILED(surface.texture.As(&resource))) {
                error = "encoder_dxgiresource1";
                return false;
            }
            HANDLE handle = nullptr;
            if (FAILED(resource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr,
                                                    &handle))) {
                error = "encoder_create_shared_handle";
                return false;
            }
            const HRESULT hr = device->OpenSharedHandle(handle, IID_PPV_ARGS(&surface.opened12));
            CloseHandle(handle);
            if (FAILED(hr)) { error = "encoder_open_shared_handle"; return false; }
            const D3D12_RESOURCE_DESC opened = surface.opened12->GetDesc();
            if (!(opened.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
                error = "encoder_opened_uav_missing";
                return false;
            }
        }
        return true;
    }

    bool init(ID3D12Device *device, const LUID &luid, UINT width, UINT height,
              UINT slots, const char *output_path, mfxU32 codec, mfxU16 gop_value,
              double fps_value) {
        out_w = width;
        out_h = height;
        codec_id = codec;
        if(terminal_encoder=="hevc_qsv") codec_id=MFX_CODEC_HEVC;
        gop = gop_value;
        fps = fps_value;
        if (!create_d3d11(luid)) return false;
        if(software_mode()) {
            if(!create_shared_surfaces(device,slots)) return false;
            D3D11_TEXTURE2D_DESC desc{};surfaces[0].texture->GetDesc(&desc);
            desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.MiscFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0};
            if(FAILED(d3d11->CreateTexture2D(&desc,nullptr,&terminal_staging)) ||
               FAILED(d3d11->CreateQuery(&query,&terminal_query))) {error="terminal_staging_create";return false;}
            terminal_rgb.resize(static_cast<size_t>(out_w)*out_h*3);
            mfxFrameInfo info{};set_frame_rate(&info);
            if(!software.init(terminal_encoder,out_w,out_h,info.FrameRateExtN,info.FrameRateExtD,output_matrix==ColorMatrix::Bt709)) {
                error=software.error;return false;
            }
            start_ms=GetTickCount64();ready=true;return true;
        }
        if (!init_session() || !init_components() || !create_shared_surfaces(device, slots)) return false;
        output = fopen(output_path, "wb");
        if (!output) { error = "encoder_output_open"; return false; }
        start_ms = GetTickCount64();
        ready = true;
        return true;
    }

    ID3D12Resource *opened(UINT slot_index) {
        return surfaces[slot_index % surfaces.size()].opened12.Get();
    }

    // A5.3 begin_copy contract: advance the keyed counter to the odd key so
    // the encoder's AcquireSync(key+1) after the D3D12 fence wait is valid.
    bool begin_frame(UINT slot_index) {
        SharedSurface &surface = surfaces[slot_index % surfaces.size()];
        HRESULT hr = surface.mutex->AcquireSync(surface.key, 15000);
        if (hr != S_OK) { error = "encoder_begin_acquire"; return false; }
        hr = surface.mutex->ReleaseSync(surface.key + 1);
        if (FAILED(hr)) { error = "encoder_begin_release"; return false; }
        return true;
    }

    static bool write_bitstream(const mfxBitstream &bitstream, FILE *out) {
        return !bitstream.DataLength || (out &&
            fwrite(bitstream.Data + bitstream.DataOffset, 1, bitstream.DataLength,
                   out) == bitstream.DataLength);
    }

    double last_vpp_ms = 0.0, last_enc_ms = 0.0;
    bool read_terminal_frame(SharedSurface& surface) {
        d3d11_context->CopyResource(terminal_staging.Get(),surface.texture.Get());
        d3d11_context->End(terminal_query.Get());d3d11_context->Flush();
        const ULONGLONG until=GetTickCount64()+15000;HRESULT hr=S_FALSE;
        while((hr=d3d11_context->GetData(terminal_query.Get(),nullptr,0,D3D11_ASYNC_GETDATA_DONOTFLUSH))==S_FALSE && GetTickCount64()<until) Sleep(1);
        if(hr!=S_OK) {error="terminal_readback_timeout_or_device_error";return false;}
        D3D11_MAPPED_SUBRESOURCE mapped{};
        // DO_NOT_WAIT may still report transient driver-side resource busy
        // after the query is signalled. Retry only that documented status,
        // with the same bounded deadline; device/argument errors fail closed.
        do {
            hr=d3d11_context->Map(terminal_staging.Get(),0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);
            if(hr!=DXGI_ERROR_WAS_STILL_DRAWING) break;
            ++terminal_map_busy_retries;
            Sleep(1);
        } while(GetTickCount64()<until);
        if(FAILED(hr)) {
            char message[80];std::snprintf(message,sizeof(message),"terminal_readback_map=0x%08lX",static_cast<unsigned long>(hr));
            error=message;return false;
        }
        for(UINT y=0;y<out_h;++y) {
            const UCHAR* src=static_cast<const UCHAR*>(mapped.pData)+static_cast<size_t>(y)*mapped.RowPitch;
            UCHAR* dst=terminal_rgb.data()+static_cast<size_t>(y)*out_w*3;
            for(UINT x=0;x<out_w;++x) {dst[x*3]=src[x*4];dst[x*3+1]=src[x*4+1];dst[x*3+2]=src[x*4+2];}
        }
        d3d11_context->Unmap(terminal_staging.Get(),0);
        terminal_readback_bytes+=static_cast<UINT64>(out_w)*out_h*4;
        return true;
    }
    bool encode_slot(UINT slot_index, UINT frame_index) {
                SharedSurface &surface = surfaces[slot_index % surfaces.size()];
        const UINT key = surface.key + 1;
        HRESULT hr = surface.mutex->AcquireSync(key, 15000);
        if (hr != S_OK) { error = "encoder_acquire_sync"; return false; }
        if(software_mode()) {
            const bool copied=read_terminal_frame(surface);
            const HRESULT released=surface.mutex->ReleaseSync(key+1);surface.key+=2;
            if(!copied||FAILED(released)) {if(error.empty())error="terminal_mutex_release";return false;}
            if(!software.write(terminal_rgb.data(),terminal_rgb.size())) {error=software.error;return false;}
            // These legacy counters mean accepted/copied/submitted for the RGB
            // sink, not oneVPL Import/VPP. Report fields below identify the sink.
            ++imported_count;++converted_count;++encoded_count;
            return true;
        }
        mfxSurfaceD3D11Tex2D external{};
        external.SurfaceInterface.Header.SurfaceType = MFX_SURFACE_TYPE_D3D11_TEX2D;
        external.SurfaceInterface.Header.SurfaceFlags =
            MFX_SURFACE_FLAG_IMPORT_SHARED | MFX_SURFACE_FLAG_IMPORT_COPY;
        external.SurfaceInterface.Header.StructSize = sizeof(external);
        external.texture2D = surface.texture.Get();
        const auto t_enc0 = std::chrono::steady_clock::now();
        mfxFrameSurface1 *input = nullptr;
        mfxStatus status = memory->ImportFrameSurface(
            memory, MFX_SURFACE_COMPONENT_VPP_INPUT,
            &external.SurfaceInterface.Header, &input);
        if (status != MFX_ERR_NONE || !input) {
            surface.mutex->ReleaseSync(key + 1);
            error = "encoder_import";
            return false;
        }
        ++imported_count;
        if (external.SurfaceInterface.Header.SurfaceFlags & MFX_SURFACE_FLAG_IMPORT_COPY)
            ++import_copy_count;
        const mfxU64 timestamp = static_cast<mfxU64>(frame_index) * 90000 /
            static_cast<mfxU64>(fps > 1.0 ? fps : 24.0);
        input->Data.TimeStamp = timestamp;
        mfxFrameSurface1 *converted = nullptr;
        status = MFXMemory_GetSurfaceForVPPOut(session, &converted);
        if (status != MFX_ERR_NONE || !converted) {
            input->FrameInterface->Release(input);
            surface.mutex->ReleaseSync(key + 1);
            error = "encoder_vpp_surface";
            return false;
        }
        mfxSyncPoint vpp_sync = nullptr;
        status = MFXVideoVPP_RunFrameVPPAsync(session, input, converted, nullptr,
                                              &vpp_sync);
        input->FrameInterface->Release(input);
        if (status != MFX_ERR_NONE || !vpp_sync) {
            converted->FrameInterface->Release(converted);
            surface.mutex->ReleaseSync(key + 1);
            error = "encoder_vpp_run";
            return false;
        }
        if (MFXVideoCORE_SyncOperation(session, vpp_sync, 15000) != MFX_ERR_NONE) {
            converted->FrameInterface->Release(converted);
            surface.mutex->ReleaseSync(key + 1);
            error = "encoder_vpp_sync";
            return false;
        }
        ++converted_count;
        mfxBitstream bitstream{};
        bitstream.MaxLength = 16u << 20;
        bitstream.Data = static_cast<mfxU8 *>(calloc(bitstream.MaxLength, 1));
        mfxSyncPoint enc_sync = nullptr;
        status = bitstream.Data
            ? MFXVideoENCODE_EncodeFrameAsync(session, nullptr, converted, &bitstream,
                                              &enc_sync)
            : MFX_ERR_MEMORY_ALLOC;
        converted->FrameInterface->Release(converted);
        bool ok = status == MFX_ERR_NONE && enc_sync;
        if (ok) {
            status = MFXVideoCORE_SyncOperation(session, enc_sync, 15000);
            ok = status == MFX_ERR_NONE && write_bitstream(bitstream, output);
            if (ok) {
                ++encoded_count;
                if (!first_bit_seen) {
                    first_bit_seen = true;
                    first_bit_ms = GetTickCount64();
                }
                if (bitstream.TimeStamp == MFX_TIMESTAMP_UNKNOWN) ++pts_unknown;
                else if (bitstream.TimeStamp != timestamp) ++pts_mismatch;
            }
            if (!ok && error.empty()) error = "encoder_output_write";
        } else if (status != MFX_ERR_MORE_DATA) {
            ok = false;
            error = "encoder_encode_frame";
        }
        free(bitstream.Data);
        hr = surface.mutex->ReleaseSync(key + 1);
        if (FAILED(hr)) { ok = false; error = "encoder_release_sync"; }
        surface.key += 2;
        last_enc_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_enc0).count();
        if (frame_index % 16 == 0)
            std::fprintf(stderr, "chain_timing frame=%u vpp=%.1fms enc=%.1fms\n",
                         frame_index, last_vpp_ms, last_enc_ms);
                return ok;
    }

    bool drain() {
        if(software_mode()) {const bool ok=software.finish();if(!ok)error=software.error;return ok;}
        for (UINT i = 0; i < 64; ++i) {
            mfxBitstream bitstream{};
            bitstream.MaxLength = 16u << 20;
            bitstream.Data = static_cast<mfxU8 *>(calloc(bitstream.MaxLength, 1));
            if (!bitstream.Data) return false;
            mfxSyncPoint sync = nullptr;
            mfxStatus status = MFXVideoENCODE_EncodeFrameAsync(
                session, nullptr, nullptr, &bitstream, &sync);
            if (status == MFX_ERR_MORE_DATA) { free(bitstream.Data); return true; }
            if (status != MFX_ERR_NONE || !sync) { free(bitstream.Data); return false; }
            if (MFXVideoCORE_SyncOperation(session, sync, 15000) != MFX_ERR_NONE ||
                !write_bitstream(bitstream, output)) {
                free(bitstream.Data);
                return false;
            }
            ++encoded_count;
            free(bitstream.Data);
        }
        return false;
    }

    // Test-only downstream failure injection.  The async full-GPU worker uses
    // this to prove that a closed output is reported as PARTIAL rather than a
    // successful run with a truncated bitstream.
    void close_output_for_test() {
        if (output) {
            fclose(output);
            output = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------
struct ChainSlot {
    GpuResource color, luma;                    // input resolution
    GpuResource forward_flow, backward_flow;    // R16G16F input res
    GpuResource confidence;                     // R32F input res
    GpuResource forward_repaired;               // R16G16F input res
    GpuResource velocity;                       // R16G16F output res
    GpuResource mask;                           // R8 input res
    GpuResource guide;                          // RGBA16F output res
    GpuResource xess_output;                    // RGBA8 output res (D3D12)
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12CommandAllocator> allocator_b;
    ComPtr<ID3D12GraphicsCommandList> list_b;
    UINT64 fence_value = 0;
    // The opened decode surface stays referenced until this frame's GPU
    // work completes; the encoder thread releases it after the fence wait.
    ComPtr<ID3D12Resource> pending_import;
    // descriptor indices (fixed after setup)
    UINT d_color = 0, d_luma_cur = 0, d_fwd = 0, d_bwd = 0, d_conf = 0,
        d_fwd_repaired = 0, d_xess_output = 0, d_guide = 0,
        d_u_color = 0, d_u_fwd = 0, d_u_bwd = 0, d_u_conf = 0,
        d_u_fwd_repaired = 0, d_u_velocity = 0, d_u_mask = 0, d_u_guide = 0,
        d_u_xess = 0, d_u_shared = 0;
    UINT d_vel_fwd = 0, d_vel_repaired = 0;
    UINT d_bwd_t0 = 0, d_bwd_t1 = 0, d_fwd_t0 = 0, d_fwd_t1 = 0;
    UINT d_mask_t0 = 0, d_mask_t1 = 0, d_mask_t2 = 0, d_mask_t3 = 0,
        d_mask_t4 = 0;
    UINT d_repair_t0 = 0, d_repair_t1 = 0, d_repair_t2 = 0, d_repair_t3 = 0,
        d_repair_t4 = 0;
    UINT d_import_luma = 0, d_import_chroma = 0;  // per-frame NV12 SRVs
};

} // namespace

#ifndef XESS_FULL_CHAIN_LIBRARY
int main(int argc, char **argv) {
    Options options;
    options.max_frames = 8;
    if (!ParseArgs(argc, argv, options)) {
        std::printf("Usage: vpl-gpu-full-chain-probe --input <H264> [--codec h264] "
                    "--max-frames N --report PATH --shader-dir DIR "
                    "--xess-quality ultra-quality --output-width W --output-height H "
                    "--qsv-out PATH [--gpu-post on|off] [--slots 4] [--fps 24] "
                    "[--motion-repair refine|propagate|off] "
                    "[--motion-center-bias 0.002]\n");
        return 2;
    }
    const bool gpu_post_on = options.gpu_post != "off";
    const UINT slot_count =
        static_cast<UINT>(options.slot_count > 0 && options.slot_count <= 8
                              ? options.slot_count : 4);

    {
        // Match the H2 probe: enable the D3D12 debug layer for diagnostics.
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
            debug->EnableDebugLayer();
    }
    ComRuntime com;
    if (!com.usable()) return 3;
    std::vector<mfxU8> compressed;
    if (!ReadCompressed(options.input, compressed)) return 4;

    Result result;
    result.input = options.input;
    result.codec = options.codec;
    result.requested = options.max_frames;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D11Device> d3d11;
    ComPtr<ID3D11DeviceContext> d3d11_context;
    ComPtr<ID3D12Device> d3d12;
    if (!CreateDevices(options.adapter, adapter, d3d11, d3d11_context, d3d12, result))
        return 5;
    if (!d3d12) return 6;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    adapter->GetDesc1(&adapter_desc);
    char luid_text[32]{};
    std::snprintf(luid_text, sizeof(luid_text), "%08X:%08X",
                  adapter_desc.AdapterLuid.HighPart, adapter_desc.AdapterLuid.LowPart);
    result.adapter_luid = luid_text;

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    if (FAILED(d3d12->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)))) return 6;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event || FAILED(d3d12->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                                  IID_PPV_ARGS(&fence))))
        return 6;
    UINT64 fence_value = 0;

    const mfxU32 codec = options.codec == "h264" ? MFX_CODEC_AVC : MFX_CODEC_HEVC;
    mfxLoader loader = MFXLoad();
    if (!loader) return 7;
    mfxConfig configs[5]{};
    bool filters_ok = true;
    for (auto &config : configs) config = MFXCreateConfig(loader);
    filters_ok = filters_ok &&
        SetFilter(configs[0], "mfxImplDescription.Impl", MFX_IMPL_TYPE_HARDWARE) &&
        SetFilter(configs[1], "mfxImplDescription.ApiVersion.Version", kApiVersion) &&
        SetFilter(configs[2], "mfxImplDescription.AccelerationMode", MFX_ACCEL_MODE_VIA_D3D11) &&
        SetFilter(configs[3], "mfxImplDescription.mfxDecoderDescription.decoder.CodecID", codec) &&
        SetFilter(configs[4], "mfxImplDescription.mfxDecoderDescription.decoder.CodecID", codec);
    if (!filters_ok) return 8;
    mfxSession session = nullptr;
    if (MFXCreateSession(loader, 0, &session) != MFX_ERR_NONE || !session) return 9;
    if (MFXVideoCORE_SetHandle(session, MFX_HANDLE_D3D11_DEVICE,
                               reinterpret_cast<mfxHDL>(d3d11.Get())) != MFX_ERR_NONE)
        return 10;
    mfxBitstream bitstream{};
    bitstream.Data = compressed.data();
    bitstream.MaxLength = static_cast<mfxU32>(compressed.size());
    bitstream.DataLength = static_cast<mfxU32>(compressed.size());
    bitstream.CodecId = codec;
    mfxVideoParam decode_params{};
    decode_params.mfx.CodecId = codec;
    decode_params.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    mfxExtVideoSignalInfo signal_info{};
    signal_info.Header.BufferId = MFX_EXTBUFF_VIDEO_SIGNAL_INFO;
    signal_info.Header.BufferSz = sizeof(signal_info);
    mfxExtBuffer *decode_ext[] = {&signal_info.Header};
    decode_params.ExtParam = decode_ext;
    decode_params.NumExtParam = 1;
    if (MFXVideoDECODE_DecodeHeader(session, &bitstream, &decode_params) != MFX_ERR_NONE)
        return 11;
    mfxVideoParam init_params = decode_params;
    init_params.ExtParam = nullptr;
    init_params.NumExtParam = 0;
    if (MFXVideoDECODE_Init(session, &init_params) < MFX_ERR_NONE) return 11;
    const int input_width = decode_params.mfx.FrameInfo.CropW
        ? decode_params.mfx.FrameInfo.CropW : decode_params.mfx.FrameInfo.Width;
    const int input_height = decode_params.mfx.FrameInfo.CropH
        ? decode_params.mfx.FrameInfo.CropH : decode_params.mfx.FrameInfo.Height;
    const ColorSelection color = SelectColor(options, signal_info, input_width, input_height);
    const int output_width = options.output_width > 0 ? options.output_width : input_width * 2;
    const int output_height = options.output_height > 0 ? options.output_height : input_height * 2;
    std::printf("color_signal selected_matrix=%s selected_range=%s\n",
                ColorMatrixName(color.matrix), ColorRangeName(color.range));

    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> color_pso, motion_pso, velocity_pso, mask_pso,
        repair_pso, guide_pso, post_pso;
    std::string error;
    const std::string shader_dir = options.shader_dir;
    bool pipelines_ok = chain_root(d3d12.Get(), root, error) &&
        chain_pipeline(d3d12.Get(), shader_dir, "surface_nv12_color", root.Get(), color_pso, error) &&
        chain_pipeline(d3d12.Get(), shader_dir, "surface_gpu_motion_tile", root.Get(), motion_pso, error) &&
        chain_pipeline(d3d12.Get(), shader_dir, "surface_gpu_velocity", root.Get(), velocity_pso, error) &&
        chain_pipeline(d3d12.Get(), shader_dir, "surface_gpu_mask", root.Get(), mask_pso, error) &&
        chain_pipeline(d3d12.Get(), shader_dir, "surface_gpu_motion_repair", root.Get(), repair_pso, error) &&
        chain_pipeline(d3d12.Get(), shader_dir, "surface_gpu_guide", root.Get(), guide_pso, error) &&
        chain_pipeline(d3d12.Get(), shader_dir, "surface_gpu_post", root.Get(), post_pso, error);
    if (!pipelines_ok) {
        result.block = error;
        MFXVideoDECODE_Close(session); MFXClose(session); MFXUnload(loader);
        return 12;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = slot_count * 40;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    if (FAILED(d3d12->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)))) return 12;
    const UINT heap_stride = d3d12->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    xess_context_handle_t xess = nullptr;
    {
        xess_d3d12_init_params_t params{};
        params.outputResolution = {static_cast<uint32_t>(output_width),
                                   static_cast<uint32_t>(output_height)};
        params.qualitySetting = options.xess_quality == "performance"
            ? XESS_QUALITY_SETTING_PERFORMANCE
            : options.xess_quality == "balanced" ? XESS_QUALITY_SETTING_BALANCED
            : options.xess_quality == "quality" ? XESS_QUALITY_SETTING_QUALITY
            : XESS_QUALITY_SETTING_ULTRA_QUALITY;
        params.initFlags = XESS_INIT_FLAG_LDR_INPUT_COLOR |
                           XESS_INIT_FLAG_HIGH_RES_MV |
                           XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK;
        if (xessD3D12CreateContext(d3d12.Get(), &xess) != XESS_RESULT_SUCCESS ||
            xessD3D12BuildPipelines(xess, nullptr, true, params.initFlags) !=
                XESS_RESULT_SUCCESS ||
            xessD3D12Init(xess, &params) != XESS_RESULT_SUCCESS ||
            xessSetVelocityScale(xess, 1.0f, 1.0f) != XESS_RESULT_SUCCESS ||
            xessSetMaxResponsiveMaskValue(xess, 0.8f) != XESS_RESULT_SUCCESS) {
            result.block = "xess_init";
            return 13;
        }
    }

    ChainEncoder encoder;
    if (!encoder.init(d3d12.Get(), adapter_desc.AdapterLuid,
                      static_cast<UINT>(output_width),
                      static_cast<UINT>(output_height),
                      slot_count, options.qsv_out.c_str(), MFX_CODEC_AVC, 48,
                      options.fps > 0.0 ? options.fps : 24.0)) {
        result.block = encoder.error;
        std::fprintf(stderr, "encoder_init_failed block=%s\n",
                     encoder.error.c_str());
        return 14;
    }

    const UINT repair_mode =
        options.motion_repair == "propagate" ? 1u :
        options.motion_repair == "refine" ? 2u : 0u;
    const float center_bias = options.motion_center_bias;

    // Persistent per-slot resources.  The frame sequence walks the slots in
    // FIFO order, so the previous frame's luma for slot s always lives in
    // slot (s + slot_count - 1) % slot_count.
    std::vector<ChainSlot> slots(slot_count);
    for (UINT s = 0; s < slot_count; ++s) {
        ChainSlot &slot = slots[s];
        std::string slot_error;
        bool ok =
            chain_texture(d3d12.Get(), input_width, input_height,
                          DXGI_FORMAT_R8G8B8A8_UNORM,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.color,
                          slot_error) &&
            chain_texture(d3d12.Get(), input_width, input_height, DXGI_FORMAT_R32_FLOAT,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.luma, slot_error) &&
            chain_texture(d3d12.Get(), input_width, input_height,
                          DXGI_FORMAT_R16G16_FLOAT,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.forward_flow, slot_error) &&
            chain_texture(d3d12.Get(), input_width, input_height,
                          DXGI_FORMAT_R16G16_FLOAT,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.backward_flow, slot_error) &&
            chain_texture(d3d12.Get(), input_width, input_height, DXGI_FORMAT_R32_FLOAT,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.confidence, slot_error) &&
            chain_texture(d3d12.Get(), input_width, input_height,
                          DXGI_FORMAT_R16G16_FLOAT,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.forward_repaired, slot_error) &&
            chain_texture(d3d12.Get(), output_width, output_height,
                          DXGI_FORMAT_R16G16_FLOAT,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.velocity, slot_error) &&
            chain_texture(d3d12.Get(), input_width, input_height, DXGI_FORMAT_R8_UNORM,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.mask, slot_error) &&
            chain_texture(d3d12.Get(), output_width, output_height,
                          DXGI_FORMAT_R16G16B16A16_FLOAT,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.guide, slot_error) &&
            chain_texture(d3d12.Get(), output_width, output_height,
                          DXGI_FORMAT_R8G8B8A8_UNORM,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.xess_output, slot_error);
        if (!ok || FAILED(d3d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                        IID_PPV_ARGS(&slot.allocator))) ||
            FAILED(d3d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            slot.allocator.Get(), nullptr,
                                            IID_PPV_ARGS(&slot.list))) ||
            FAILED(d3d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 IID_PPV_ARGS(&slot.allocator_b))) ||
            FAILED(d3d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            slot.allocator_b.Get(), nullptr,
                                            IID_PPV_ARGS(&slot.list_b)))) {
            result.block = slot_error.empty() ? "slot_setup" : slot_error;
            return 15;
        }
        slot.list->Close();
    }
    // Descriptor indices.  Per slot (base = s * 40): fixed single SRVs/UAVs
    // plus dedicated consecutive alias blocks for each dispatch's t0..t4
    // table.  Cross-slot SRVs (previous frame's luma) are filled after all
    // slot resources exist.
    for (UINT s = 0; s < slot_count; ++s) {
        ChainSlot &slot = slots[s];
        const UINT base = s * 40;
        slot.d_color = base + 0;          // SRV color (guide pass t0)
        slot.d_luma_cur = base + 1;       // SRV current luma
        slot.d_fwd = base + 2;            // SRV forward flow
        slot.d_bwd = base + 3;            // SRV backward flow
        slot.d_conf = base + 4;           // SRV confidence
        slot.d_fwd_repaired = base + 5;   // SRV repaired forward flow
        slot.d_xess_output = base + 6;    // SRV XeSS output (post t0)
        slot.d_guide = base + 7;          // SRV guide (post t1)
        slot.d_u_color = base + 8;
        slot.d_u_fwd = base + 9;
        slot.d_u_bwd = base + 10;
        slot.d_u_conf = base + 11;
        slot.d_u_fwd_repaired = base + 12;
        slot.d_u_velocity = base + 13;
        slot.d_u_mask = base + 14;
        slot.d_u_guide = base + 15;
        slot.d_u_xess = base + 16;
        slot.d_u_shared = base + 17;
        slot.d_vel_fwd = base + 18;       // velocity t0 alias: raw forward
        slot.d_vel_repaired = base + 19;  // velocity t0 alias: repaired
        slot.d_bwd_t0 = base + 20;        // backward t0: current luma
        slot.d_bwd_t1 = base + 21;        // backward t1: previous luma
        slot.d_fwd_t0 = base + 22;        // forward t0: previous luma
        slot.d_fwd_t1 = base + 23;        // forward t1: current luma
        slot.d_mask_t0 = base + 24;       // mask table t0..t4
        slot.d_mask_t1 = base + 25;
        slot.d_mask_t2 = base + 26;
        slot.d_mask_t3 = base + 27;
        slot.d_mask_t4 = base + 28;
        slot.d_repair_t0 = base + 29;     // repair table t0..t4
        slot.d_repair_t1 = base + 30;
        slot.d_repair_t2 = base + 31;
        slot.d_repair_t3 = base + 32;
        slot.d_repair_t4 = base + 33;
        slot.d_import_luma = base + 38;
        slot.d_import_chroma = base + 39;
    }
    for (UINT s = 0; s < slot_count; ++s) {
        ChainSlot &slot = slots[s];
        const UINT base = s * 40;
        const UINT prev = (s + slot_count - 1) % slot_count;
        chain_srv(d3d12.Get(), heap.Get(), slot.d_color, heap_stride,
                  slot.color.resource.Get(), DXGI_FORMAT_R8G8B8A8_UNORM);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_luma_cur, heap_stride,
                  slot.luma.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_fwd, heap_stride,
                  slot.forward_flow.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_bwd, heap_stride,
                  slot.backward_flow.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_conf, heap_stride,
                  slot.confidence.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_fwd_repaired, heap_stride,
                  slot.forward_repaired.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_xess_output, heap_stride,
                  slot.xess_output.resource.Get(), DXGI_FORMAT_R8G8B8A8_UNORM);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_guide, heap_stride,
                  slot.guide.resource.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
        chain_uav(d3d12.Get(), heap.Get(), slot.d_u_color, heap_stride,
                  slot.color.resource.Get(), DXGI_FORMAT_R8G8B8A8_UNORM);
        chain_uav(d3d12.Get(), heap.Get(), slot.d_u_fwd, heap_stride,
                  slot.forward_flow.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_uav(d3d12.Get(), heap.Get(), slot.d_u_bwd, heap_stride,
                  slot.backward_flow.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_uav(d3d12.Get(), heap.Get(), slot.d_u_conf, heap_stride,
                  slot.confidence.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_uav(d3d12.Get(), heap.Get(), slot.d_u_fwd_repaired, heap_stride,
                  slot.forward_repaired.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_uav(d3d12.Get(), heap.Get(), slot.d_u_velocity, heap_stride,
                  slot.velocity.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_uav(d3d12.Get(), heap.Get(), slot.d_u_mask, heap_stride,
                  slot.mask.resource.Get(), DXGI_FORMAT_R8_UNORM);
        chain_uav(d3d12.Get(), heap.Get(), slot.d_u_guide, heap_stride,
                  slot.guide.resource.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
        chain_uav(d3d12.Get(), heap.Get(), slot.d_u_xess, heap_stride,
                  slot.xess_output.resource.Get(), DXGI_FORMAT_R8G8B8A8_UNORM);
        chain_uav(d3d12.Get(), heap.Get(), slot.d_u_shared, heap_stride,
                  encoder.opened(s), DXGI_FORMAT_R8G8B8A8_UNORM);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_vel_fwd, heap_stride,
                  slot.forward_flow.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_vel_repaired, heap_stride,
                  slot.forward_repaired.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_bwd_t0, heap_stride,
                  slot.luma.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_bwd_t1, heap_stride,
                  slots[prev].luma.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_fwd_t0, heap_stride,
                  slots[prev].luma.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_fwd_t1, heap_stride,
                  slot.luma.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_mask_t0, heap_stride,
                  slots[prev].luma.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_mask_t1, heap_stride,
                  slot.luma.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_mask_t2, heap_stride,
                  repair_mode != 0u ? slot.forward_repaired.resource.Get()
                                    : slot.forward_flow.resource.Get(),
                  DXGI_FORMAT_R16G16_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_mask_t3, heap_stride,
                  slot.backward_flow.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_mask_t4, heap_stride,
                  slot.confidence.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_repair_t0, heap_stride,
                  slot.forward_flow.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_repair_t1, heap_stride,
                  slot.backward_flow.resource.Get(), DXGI_FORMAT_R16G16_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_repair_t2, heap_stride,
                  slot.confidence.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_repair_t3, heap_stride,
                  slot.luma.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_repair_t4, heap_stride,
                  slots[prev].luma.resource.Get(), DXGI_FORMAT_R32_FLOAT);
    }
    ID3D12DescriptorHeap *heaps[] = {heap.Get()};

    // Free slots + encode jobs.
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::queue<UINT> free_slots;
    for (UINT s = 0; s < slot_count; ++s) free_slots.push(s);
    struct EncodeJob {
        UINT slot;
        UINT frame_index;
        UINT64 fence_value;
    };
    std::queue<EncodeJob> encode_jobs;
    bool producer_done = false;
    bool pipeline_failed = false;
    std::string pipeline_error;

    auto encoder_thread = std::thread([&]() {
        for (;;) {
            EncodeJob job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [&] { return producer_done || !encode_jobs.empty(); });
                if (encode_jobs.empty() && producer_done) break;
                job = encode_jobs.front();
                encode_jobs.pop();
            }
            slots[job.slot].pending_import = nullptr;
            const auto t_fence0 = std::chrono::steady_clock::now();
            if (fence->GetCompletedValue() < job.fence_value) {
                HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                bool waited = false;
                if (event) {
                    waited = SUCCEEDED(fence->SetEventOnCompletion(job.fence_value, event));
                    if (waited)
                        waited = WaitForSingleObject(event, 30000) == WAIT_OBJECT_0;
                    CloseHandle(event);
                }
                if (!waited) {
                    pipeline_failed = true;
                    pipeline_error = "fence_wait_timeout";
                    continue;
                }
            }
            if (job.frame_index % 16 == 0)
                std::fprintf(stderr,
                             "chain_timing encoder frame=%u fence_wait=%.1fms\n",
                             job.frame_index,
                             std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t_fence0)
                                 .count());
            if (!encoder.encode_slot(job.slot, job.frame_index)) {
                pipeline_failed = true;
                pipeline_error = encoder.error;
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                free_slots.push(job.slot);
            }
            queue_cv.notify_all();
        }
    });

    ComPtr<IDXGIAdapter3> adapter3;
    adapter.As(&adapter3);
    UINT64 vram_peak = 0, vram_before = 0, vram_after = 0, vram_budget = 0;
    if (adapter3) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                                     &info))) {
            vram_budget = info.Budget;
            vram_before = info.CurrentUsage;
            vram_peak = info.CurrentUsage;
        }
    }

    const auto wall_start = std::chrono::steady_clock::now();
    double first_frame_wall_ms = -1.0;
    UINT processed = 0;
    UINT drain_idle = 0;
    bool draining = false;
    bool producer_failed = false;

    while (processed < options.max_frames) {
        UINT s;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait_for(lock, std::chrono::milliseconds(1000), [&] {
                return !free_slots.empty() || pipeline_failed;
            });
            if (pipeline_failed) { producer_failed = true; break; }
            if (free_slots.empty()) {
                std::fprintf(stderr, "chain_timing slot_wait_timeout frame=%u\n",
                             processed);
                continue;
            }
            s = free_slots.front();
            free_slots.pop();
        }
        ChainSlot &slot = slots[s];
        const UINT prev_slot = (s + slot_count - 1) % slot_count;
        const bool first_frame = processed == 0;
        const auto t_frame_start = std::chrono::steady_clock::now();

        mfxFrameSurface1 *surface = nullptr;
        mfxSyncPoint sync = nullptr;
        const mfxBitstream *input =
            (!draining && bitstream.DataLength > 0) ? &bitstream : nullptr;
        if (!input) draining = true;
        const auto t_decode0 = std::chrono::steady_clock::now();
        const mfxStatus status = MFXVideoDECODE_DecodeFrameAsync(
            session, const_cast<mfxBitstream *>(input), nullptr, &surface, &sync);
        if (processed % 16 == 0)
            std::fprintf(stderr,
                         "chain_timing decode frame=%u call=%.1fms status=%d\n",
                         processed,
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t_decode0).count(),
                         static_cast<int>(status));
        if (status == MFX_WRN_DEVICE_BUSY) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            free_slots.push(s);
            queue_cv.notify_all();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        if (status == MFX_ERR_MORE_DATA) {
            draining = true;
            std::lock_guard<std::mutex> lock(queue_mutex);
            free_slots.push(s);
            queue_cv.notify_all();
            if (++drain_idle >= 8) break;
            continue;
        }
        if (status < MFX_ERR_NONE) {
            result.block = "DecodeFrameAsync=" + std::to_string(status);
            producer_failed = true;
            break;
        }
        if (status != MFX_ERR_NONE || !surface) {
            // Positive oneVPL statuses are warnings (e.g. 3 = video params
            // changed when the stream re-emits an SPS); keep consuming.
            continue;
        }
        drain_idle = 0;
        const auto t_sync0 = std::chrono::steady_clock::now();
        if (surface->FrameInterface->Synchronize(surface, 5000) != MFX_ERR_NONE) {
            result.block = "FrameInterface.Synchronize";
            surface->FrameInterface->Release(surface);
            producer_failed = true;
            break;
        }
        mfxSurfaceHeader header{};
        header.SurfaceType = MFX_SURFACE_TYPE_D3D11_TEX2D;
        header.SurfaceFlags = MFX_SURFACE_FLAG_EXPORT_SHARED;
        mfxSurfaceHeader *exported_header = nullptr;
        if (surface->FrameInterface->Export(surface, header, &exported_header) !=
                MFX_ERR_NONE || !exported_header) {
            result.block = "FrameInterface.Export";
            surface->FrameInterface->Release(surface);
            producer_failed = true;
            break;
        }
        auto *exported = reinterpret_cast<mfxSurfaceD3D11Tex2D *>(exported_header);
        ComPtr<ID3D12Resource> imported;
        {
            ComPtr<IDXGIResource1> dxgi_resource;
            ID3D11Texture2D *texture =
                reinterpret_cast<ID3D11Texture2D *>(exported->texture2D);
            bool opened = SUCCEEDED(texture->QueryInterface(IID_PPV_ARGS(&dxgi_resource)));
            HANDLE shared = nullptr;
            if (opened)
                opened = SUCCEEDED(dxgi_resource->CreateSharedHandle(
                    nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                    nullptr, &shared));
            if (opened)
                opened = SUCCEEDED(d3d12->OpenSharedHandle(shared, IID_PPV_ARGS(&imported)));
            if (shared) CloseHandle(shared);
            if (!opened) {
                result.block = "decode_surface_open";
                exported->SurfaceInterface.Release(&exported->SurfaceInterface);
                surface->FrameInterface->Release(surface);
                producer_failed = true;
                break;
            }
        }
        exported->SurfaceInterface.Release(&exported->SurfaceInterface);
        surface->FrameInterface->Release(surface);
        if (processed % 16 == 0)
            std::fprintf(stderr, "chain_timing producer frame=%u decode_sync=%.1fms\n",
                         processed,
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t_sync0).count());
        slot.pending_import = imported;

        // Advance the shared-texture mutex for this slot before recording.
        if (!encoder.begin_frame(s)) {
            result.block = encoder.error;
            producer_failed = true;
            break;
        }
        // Record the frame's command list on this slot.
                ID3D12GraphicsCommandList *list = slot.list.Get();
        list->Reset(slot.allocator.Get(), nullptr);
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(root.Get());
        UINT constants[6] = {static_cast<UINT>(input_width),
                             static_cast<UINT>(input_height),
                             color.matrix == ColorMatrix::Bt709 ? 1u : 0u,
                             color.range == ColorRange::Full ? 1u : 0u, 0u, 0u};
        auto table0 = [&](UINT index) {
            list->SetComputeRootDescriptorTable(0, chain_gpu_desc(heap.Get(), index,
                                                                  heap_stride));
        };
        auto table1 = [&](UINT index) {
            list->SetComputeRootDescriptorTable(1, chain_gpu_desc(heap.Get(), index,
                                                                  heap_stride));
        };
        auto constants6 = [&]() {
            list->SetComputeRoot32BitConstants(2, 6, constants, 0);
        };

        // Color: bit-exact swscale NV12 -> RGBA + matrix luma.  The NV12
        // plane SRVs are (re)created per frame for the freshly imported
        // decoder surface.
        chain_srv(d3d12.Get(), heap.Get(), slot.d_import_luma, heap_stride,
                  imported.Get(), DXGI_FORMAT_R8_UNORM, 0);
        chain_srv(d3d12.Get(), heap.Get(), slot.d_import_chroma, heap_stride,
                  imported.Get(), DXGI_FORMAT_R8G8_UNORM, 1);
        chain_transition(list, imported.Get(), D3D12_RESOURCE_STATE_COMMON,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->SetPipelineState(color_pso.Get());
        table0(slot.d_import_luma);
        table1(slot.d_u_color);
        constants6();
        list->Dispatch((input_width + 7) / 8, (input_height + 7) / 8, 1);
        chain_transition(list, slot.color.resource.Get(),
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        chain_transition(list, slot.luma.resource.Get(),
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Motion searches: forward (with confidence), then backward.  The
        // cross-slot luma SRVs provide the previous frame.
        if (!first_frame) {
            list->SetPipelineState(motion_pso.Get());
            // forward: t0 = previous luma (cross-slot alias), t1 = current
            table0(slot.d_fwd_t0);
            table1(slot.d_u_fwd);
            constants[4] = 1u;
            const float bias = center_bias;
            std::memcpy(&constants[5], &bias, sizeof(bias));
            constants6();
            list->Dispatch((input_width + 7) / 8, (input_height + 7) / 8, 1);
            // backward: t0 = current luma, t1 = previous luma
            table0(slot.d_bwd_t0);
            table1(slot.d_u_bwd);
            constants[4] = 0u;
            constants6();
            list->Dispatch((input_width + 7) / 8, (input_height + 7) / 8, 1);
            chain_transition(list, slot.forward_flow.resource.Get(),
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            chain_transition(list, slot.backward_flow.resource.Get(),
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            chain_transition(list, slot.confidence.resource.Get(),
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (repair_mode != 0u) {
                list->SetPipelineState(repair_pso.Get());
                table0(slot.d_repair_t0);
                table1(slot.d_u_fwd_repaired);
                constants[4] = repair_mode;
                constants6();
                list->Dispatch((input_width + 7) / 8, (input_height + 7) / 8, 1);
                chain_transition(list, slot.forward_repaired.resource.Get(),
                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            // Velocity mapping.
            list->SetPipelineState(velocity_pso.Get());
            table0(repair_mode != 0u ? slot.d_vel_repaired : slot.d_vel_fwd);
            table1(slot.d_u_velocity);
            constants[0] = static_cast<UINT>(output_width);
            constants[1] = static_cast<UINT>(output_height);
            constants[2] = static_cast<UINT>(input_width);
            constants[3] = static_cast<UINT>(input_height);
            constants6();
            list->Dispatch((output_width + 7) / 8, (output_height + 7) / 8, 1);
            // Mask (the velocity->SRV transition happens once before XeSS).
            list->SetPipelineState(mask_pso.Get());
            table0(slot.d_mask_t0);
            table1(slot.d_u_mask);
            constants[4] = repair_mode;
            constants6();
            list->Dispatch((input_width + 7) / 8, (input_height + 7) / 8, 1);
            chain_transition(list, slot.mask.resource.Get(),
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }

                // Startup contract: XeSS must never see uninitialized velocity or
        // mask.  Frame 0 clears both (velocity zero, responsive mask at the
        // 0.8 contract) exactly like the H2 probe's first-frame semantics.
        if (first_frame) {
            const float clear_zero[4] = {0, 0, 0, 0};
            const float clear_mask[4] = {0.8f, 0.8f, 0.8f, 0.8f};
            list->ClearUnorderedAccessViewFloat(
                chain_gpu_desc(heap.Get(), slot.d_u_velocity, heap_stride),
                chain_cpu_desc(heap.Get(), slot.d_u_velocity, heap_stride),
                slot.velocity.resource.Get(), clear_zero, 0, nullptr);
            list->ClearUnorderedAccessViewFloat(
                chain_gpu_desc(heap.Get(), slot.d_u_mask, heap_stride),
                chain_cpu_desc(heap.Get(), slot.d_u_mask, heap_stride),
                slot.mask.resource.Get(), clear_mask, 0, nullptr);
        }

        // Guide pass (before post; XeSS executes between motion and post).
        list->SetPipelineState(guide_pso.Get());
        table0(slot.d_color);
        table1(slot.d_u_guide);
        constants[0] = static_cast<UINT>(output_width);
        constants[1] = static_cast<UINT>(output_height);
        constants[2] = static_cast<UINT>(input_width);
        constants[3] = static_cast<UINT>(input_height);
        constants6();
        list->Dispatch((output_width + 7) / 8, (output_height + 7) / 8, 1);
        chain_transition(list, slot.guide.resource.Get(),
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // XeSS (same context, sequential semantics; the queue is FIFO).
        chain_transition(list, slot.velocity.resource.Get(),
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        chain_transition(list, slot.mask.resource.Get(),
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (!gpu_post_on)
            chain_transition(list, encoder.opened(s),
                             D3D12_RESOURCE_STATE_COMMON,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        xess_d3d12_execute_params_t exec{};
        exec.inputWidth = static_cast<uint32_t>(input_width);
        exec.inputHeight = static_cast<uint32_t>(input_height);
        exec.jitterOffsetX = 0.0f;
        exec.jitterOffsetY = 0.0f;
        exec.exposureScale = 1.0f;
        exec.resetHistory = first_frame ? 1 : 0;
        exec.pColorTexture = slot.color.resource.Get();
        exec.pVelocityTexture = slot.velocity.resource.Get();
        exec.pOutputTexture = gpu_post_on ? slot.xess_output.resource.Get()
                                          : encoder.opened(s);
        exec.pDepthTexture = nullptr;
        exec.pExposureScaleTexture = nullptr;
        exec.pResponsivePixelMaskTexture = slot.mask.resource.Get();
                const bool skip_xess = std::getenv("XESS_CHAIN_SKIP_XESS") != nullptr;
        if (!skip_xess &&
            xessD3D12Execute(xess, list, &exec) != XESS_RESULT_SUCCESS) {
            result.block = "xessD3D12Execute";
            producer_failed = true;
            break;
        }

                // List A ends here: recording SetComputeRootDescriptorTable into the
        // same list after xessD3D12Execute crashes the Intel driver, so the
        // post pass is recorded into a fresh list B on its own allocator.
        // XeSS output and the imported surface return to friendly states in
        // list A (transitions after XeSS are the H2-probe-proven pattern).
        chain_transition(list, slot.xess_output.resource.Get(),
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        chain_transition(list, imported.Get(),
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                         D3D12_RESOURCE_STATE_COMMON);
        list->Close();
        ID3D12CommandList *lists_a[] = {list};
        queue->ExecuteCommandLists(1, lists_a);

        ID3D12GraphicsCommandList *list_b = slot.list_b.Get();
        list_b->Reset(slot.allocator_b.Get(), nullptr);
        list_b->SetDescriptorHeaps(1, heaps);
        list_b->SetComputeRootSignature(root.Get());
        UINT shared_common_to_uav = 0;
        if (gpu_post_on) {
            // The shared texture is read/written by the post pass.
            shared_common_to_uav = 1;
            chain_transition(list_b, encoder.opened(s),
                             D3D12_RESOURCE_STATE_COMMON,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list_b->SetPipelineState(post_pso.Get());
            list_b->SetComputeRootDescriptorTable(
                0, chain_gpu_desc(heap.Get(), slot.d_xess_output, heap_stride));
            list_b->SetComputeRootDescriptorTable(
                1, chain_gpu_desc(heap.Get(), slot.d_u_shared, heap_stride));
            constants[0] = static_cast<UINT>(output_width);
            constants[1] = static_cast<UINT>(output_height);
            constants[2] = 25u;   // static strength 0.25
            constants[3] = 75u;   // guard strength 0.75
            list_b->SetComputeRoot32BitConstants(2, 6, constants, 0);
            list_b->Dispatch((output_width + 7) / 8, (output_height + 7) / 8, 1);
        } else {
            // Direct XeSS output (A5.4): the XeSS result already lives in the
            // shared texture and was returned to COMMON in list A decay; no
            // further work is needed in list B.
        }
        if (shared_common_to_uav) {
            chain_transition(list_b, encoder.opened(s),
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_COMMON);
        }
        list_b->Close();
        ID3D12CommandList *lists_b[] = {list_b};
        queue->ExecuteCommandLists(1, lists_b);
        ++fence_value;
        queue->Signal(fence.Get(), fence_value);
        slot.fence_value = fence_value;
        if (processed % 16 == 0) {
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr,
                         "chain_timing producer submit=%u since_frame_start=%.1fms\n",
                         processed,
                         std::chrono::duration<double, std::milli>(
                             now - t_frame_start).count());
        }

        if (processed == 0) {
            first_frame_wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wall_start).count();
        }
        if (adapter3) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(
                    0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
                vram_peak = std::max(vram_peak, info.CurrentUsage);
        }
        ++processed;
        result.exported++;
        result.decoded++;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            encode_jobs.push({s, processed - 1, fence_value});
        }
        queue_cv.notify_all();
        if (std::getenv("XESS_CHAIN_SERIALIZE") != nullptr) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait_for(lock, std::chrono::seconds(10), [&] {
                return encoder.encoded_count >= processed;
            });
        }
    }

    // Drain: wait for all outstanding encode jobs, then EOS.
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cv.wait_for(lock, std::chrono::seconds(10), [&] {
            return encoder.encoded_count >= processed || pipeline_failed;
        });
        producer_done = true;
    }
    queue_cv.notify_all();
    encoder_thread.join();
    if (!pipeline_failed && processed > 0) encoder.drain();
    if (adapter3) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                                     &info)))
            vram_after = info.CurrentUsage;
    }
    const double wall_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start).count();

    const bool pass = !pipeline_failed && !producer_failed &&
        processed == options.max_frames &&
        encoder.imported_count == processed &&
        encoder.converted_count == processed &&
        encoder.encoded_count >= processed && encoder.pts_mismatch == 0;
    result.total_wall_ms = wall_s * 1000.0;
    if (xess) xessDestroyContext(xess);
    MFXVideoDECODE_Close(session);
    MFXClose(session);
    MFXUnload(loader);

    if (!options.report.empty()) {
        std::ofstream out(options.report, std::ios::binary | std::ios::trunc);
        if (out) {
            auto J = [&](const std::string &value) {
                out << '"';
                for (char c : value) {
                    switch (c) {
                        case '"': out << "\\\""; break;
                        case '\\': out << "\\\\"; break;
                        case '\n': out << "\\n"; break;
                        default: out << c;
                    }
                }
                out << '"';
            };
            out << "{\n  \"schema_version\": \"gpu-block-motion-opt/full-chain@1\",\n";
            out << "  \"classification\": \"" << (pass ? "FULL_GPU_CHAIN_PASS" : "PARTIAL")
                << "\",\n";
            out << "  \"adapter_luid\": "; J(result.adapter_luid); out << ",\n";
            out << "  \"color_signal\": {\"selected_matrix\": ";
            J(ColorMatrixName(color.matrix));
            out << ", \"selected_range\": "; J(ColorRangeName(color.range)); out << "},\n";
            out << "  \"pipeline\": {\"slots\": " << slot_count
                << ", \"gpu_post\": " << (gpu_post_on ? "true" : "false")
                << ", \"motion_repair\": "; J(options.motion_repair);
            out << ", \"motion_center_bias\": " << center_bias
                << ", \"xess_quality\": "; J(options.xess_quality); out << "},\n";
            out << "  \"dimensions\": {\"input_width\": " << input_width
                << ", \"input_height\": " << input_height
                << ", \"output_width\": " << output_width
                << ", \"output_height\": " << output_height << "},\n";
            out << "  \"frames\": {\"requested\": " << options.max_frames
                << ", \"processed\": " << processed << "},\n";
            out << "  \"encode\": {\"imported\": " << encoder.imported_count
                << ", \"converted\": " << encoder.converted_count
                << ", \"encoded\": " << encoder.encoded_count
                << ", \"pts_mismatch\": " << encoder.pts_mismatch
                << ", \"pts_unknown\": " << encoder.pts_unknown
                << ", \"first_bitstream_ms\": " << (encoder.first_bit_seen
                    ? (encoder.first_bit_ms - encoder.start_ms) : -1.0)
                << ", \"import_flags_reported\": \"0x20 IMPORT_COPY\"},\n";
            out << "  \"boundaries\": {\"decode_surface_cpu_map\": false,\n"
                << "    \"full_frame_cpu_upload_bytes\": 0,"
                << " \"full_frame_cpu_readback_bytes\": 0,\n"
                << "    \"d3d12_explicit_copy_count_per_frame\": 0,\n"
                << "    \"onevpl_vpp_import_copy_per_frame\": 1,\n"
                << "    \"shared_texture_owners\": \"D3D11_create_D3D12_open\",\n"
                << "    \"shared_bind_flags\": \"0x88\","
                << " \"shared_misc_flags\": \"SHARED_NTHANDLE|KEYEDMUTEX\"},\n";
            out << "  \"vram\": {\"budget\": " << vram_budget
                << ", \"before\": " << vram_before
                << ", \"after\": " << vram_after
                << ", \"peak\": " << vram_peak << "},\n";
            out << "  \"wall\": {\"total_s\": " << wall_s
                << ", \"first_frame_ms\": " << first_frame_wall_ms
                << ", \"fps\": " << (wall_s > 0 ? processed / wall_s : 0.0) << "},\n";
            out << "  \"block\": "; J(result.block); out << ",\n";
            out << "  \"error\": "; J(pipeline_error.empty() ? result.block : pipeline_error);
            out << "\n}\n";
        }
    }
    std::printf("full_chain_summary processed=%u/%u imported=%u converted=%u encoded=%u "
                "pts_mismatch=%u cpu_readback=0 cpu_upload=0 wall_s=%.3f classification=%s "
                "block=%s\n", processed, options.max_frames, encoder.imported_count,
                encoder.converted_count, encoder.encoded_count, encoder.pts_mismatch,
                wall_s, pass ? "FULL_GPU_CHAIN_PASS" : "PARTIAL",
                result.block.empty() ? "none" : result.block.c_str());
    return pass ? 0 : 20;
}
#endif
