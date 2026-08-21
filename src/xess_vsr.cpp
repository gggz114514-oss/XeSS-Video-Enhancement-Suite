// xess-vsr: XeSS SR 1.2 offline/stream worker.
//
// 管线：ffmpeg 解码 -> flow.py 算 DIS 光流 -> 本程序逐帧喂给官方 libxess.dll (D3D12)
// -> 输出放大后的 rgb24 raw -> ffmpeg 编码。
//
// XeSS 配置（官方 SDK 2.1.0 ABI，见 sdk/official/inc/xess/）：
//   - xessD3D12CreateContext(device, &ctx) 只建上下文，init 参数在 xessD3D12Init 传
//   - XESS_INIT_FLAG_HIGH_RES_MV：运动矢量按输出分辨率提供（R16G16_FLOAT，半精度，
//     单位 = 输出分辨率像素/帧），免 depth 纹理
//   - XESS_INIT_FLAG_LDR_INPUT_COLOR：输入是 SDR 内容，禁用色调映射
//   - xessD3D12Execute 录制进我们自己的命令列表，随同一队列提交/等待，同步天然正确
//
// 用法:
//   xess-vsr.exe --frames frames.raw --mv mvs --in-w 854 --in-h 480
//                --out-w 1280 --out-h 720 --frames-count 243 --out out.raw
//   [--quality 0..6] [--device N] [--verbose]
// quality: 0=ultra-perf 1=perf 2=balanced 3=quality 4=ultra-quality
//          5=ultra-quality-plus 6=AA

#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <fcntl.h>
#include <io.h>

#include "xess/xess.h"
#include "xess/xess_d3d12.h"
#include "xess/xess_debug.h"
#include "shm_ring_win.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr uint32_t STREAM_FLAG_RESET = 1u << 0;
static constexpr uint32_t STREAM_FLAG_SCENE_CUT = 1u << 1;
static constexpr uint32_t STREAM_FLAG_EOS = 1u << 2;

#pragma pack(push, 1)
struct StreamHeader {
    char magic[4];
    uint16_t version;
    uint16_t headerSize;
    uint32_t frameIndex;
    uint32_t width;
    uint32_t height;
    uint32_t pixelFormat;
    uint32_t flags;
    uint32_t colorBytes;
    uint32_t motionBytes;
    uint32_t depthBytes;
    uint32_t maskBytes;
    uint32_t checksum;
};
#pragma pack(pop)
static_assert(sizeof(StreamHeader) == 48, "stream header ABI mismatch");

static uint32_t crc32_bytes(const uint8_t* data, size_t size) {
    static uint32_t table[256]{};
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t value = 0; value < 256; ++value) {
            uint32_t crc = value;
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1));
            table[value] = crc;
        }
        initialized = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i)
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

static bool read_exact(FILE* file, void* destination, size_t bytes) {
    uint8_t* output = static_cast<uint8_t*>(destination);
    size_t done = 0;
    while (done < bytes) {
        size_t count = fread(output + done, 1, bytes - done, file);
        if (!count) return false;
        done += count;
    }
    return true;
}

const char* xess_err_str(int64_t r) {
    switch (r) {
        case XESS_RESULT_SUCCESS: return "SUCCESS";
        case XESS_RESULT_WARNING_NONEXISTING_FOLDER: return "WARNING_NONEXISTING_FOLDER";
        case XESS_RESULT_WARNING_OLD_DRIVER: return "WARNING_OLD_DRIVER";
        case XESS_RESULT_ERROR_UNSUPPORTED_DEVICE: return "UNSUPPORTED_DEVICE (需 SM6.4 GPU)";
        case XESS_RESULT_ERROR_UNSUPPORTED_DRIVER: return "UNSUPPORTED_DRIVER (驱动不支持 XeSS)";
        case XESS_RESULT_ERROR_UNINITIALIZED: return "UNINITIALIZED";
        case XESS_RESULT_ERROR_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY: return "DEVICE_OUT_OF_MEMORY";
        case XESS_RESULT_ERROR_DEVICE: return "DEVICE";
        case XESS_RESULT_ERROR_NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
        case XESS_RESULT_ERROR_INVALID_CONTEXT: return "INVALID_CONTEXT";
        case XESS_RESULT_ERROR_OPERATION_IN_PROGRESS: return "OPERATION_IN_PROGRESS";
        case XESS_RESULT_ERROR_UNSUPPORTED: return "UNSUPPORTED";
        case XESS_RESULT_ERROR_CANT_LOAD_LIBRARY: return "CANT_LOAD_LIBRARY";
        case XESS_RESULT_ERROR_WRONG_CALL_ORDER: return "WRONG_CALL_ORDER";
        case XESS_RESULT_ERROR_UNKNOWN: return "UNKNOWN";
    }
    return "?";
}

static uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

// float32 -> float16（IEEE 754 舍入到最近偶数）
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF)  // inf/nan
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);   // 溢出 -> inf
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;            // 下溢 -> 0
        mant |= 0x800000u;                               // 规格化尾数
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t h = (uint32_t)exp << 10;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 0x2000u))) {
        h++;
        if (h == 0x7C00u) return (uint16_t)(sign | 0x7C00u);
    }
    return (uint16_t)(sign | h | (mant >> 13));
}

// 双线性放大运动矢量：输入 inWxInH float32 [fx,fy] -> 输出 outWxOutH float16
// （HIGH_RES_MV：单位是输出分辨率像素/帧，所以采样值要乘 out/in 缩放）
static void upsample_velocity(const float* in, uint16_t* out,
                              int inW, int inH, int outW, int outH,
                              bool nearest) {
    const float sx = (float)inW / outW;
    const float sy = (float)inH / outH;
    for (int oy = 0; oy < outH; oy++) {
        if (nearest) {
            const int iy = std::min(inH - 1, std::max(0, (int)((oy + 0.5f) * sy)));
            for (int ox = 0; ox < outW; ox++) {
                const int ix = std::min(inW - 1, std::max(0, (int)((ox + 0.5f) * sx)));
                const float* v = in + ((size_t)iy * inW + ix) * 2;
                uint16_t* o = out + ((size_t)oy * outW + ox) * 2;
                o[0] = f32_to_f16(v[0] * (float)outW / inW);
                o[1] = f32_to_f16(v[1] * (float)outH / inH);
            }
            continue;
        }
        float fy = (oy + 0.5f) * sy - 0.5f;
        int y0 = (int)fy;
        float wy = fy - y0;
        if (y0 < 0) { y0 = 0; wy = 0; }
        int y1 = y0 + 1;
        if (y1 >= inH) { y1 = inH - 1; wy = 0; }
        for (int ox = 0; ox < outW; ox++) {
            float fx = (ox + 0.5f) * sx - 0.5f;
            int x0 = (int)fx;
            float wx = fx - x0;
            if (x0 < 0) { x0 = 0; wx = 0; }
            int x1 = x0 + 1;
            if (x1 >= inW) { x1 = inW - 1; wx = 0; }
            const float* a = in + ((size_t)y0 * inW + x0) * 2;
            const float* b = in + ((size_t)y0 * inW + x1) * 2;
            const float* c = in + ((size_t)y1 * inW + x0) * 2;
            const float* d = in + ((size_t)y1 * inW + x1) * 2;
            float vx = (a[0] * (1 - wx) + b[0] * wx) * (1 - wy) + (c[0] * (1 - wx) + d[0] * wx) * wy;
            float vy = (a[1] * (1 - wx) + b[1] * wx) * (1 - wy) + (c[1] * (1 - wx) + d[1] * wx) * wy;
            uint16_t* o = out + ((size_t)oy * outW + ox) * 2;
            o[0] = f32_to_f16(vx * (float)outW / inW);
            o[1] = f32_to_f16(vy * (float)outH / inH);
        }
    }
}

struct CmdArgs {
    const char* framesRaw = nullptr;
    const char* mvDir = nullptr;
    const char* depthDir = nullptr;
    const char* maskDir = nullptr;
    const char* outRaw = nullptr;
    int inW = 0, inH = 0, outW = 0, outH = 0;
    int framesCount = 0;
    int quality = XESS_QUALITY_SETTING_QUALITY;
    int device = -1;
    bool verbose = false;
    bool nearestMv = false;
    bool stream = false;
    bool lowResMv = false;
    bool responsiveMask = false;
    float responsiveMax = 0.8f;
    const char* dumpDir = nullptr;
    const char* dbgUpload = nullptr;
    const char* resetFrames = nullptr;
    const char* shmName = nullptr;
    uint32_t shmSlots = 0;
    uint32_t shmSlotSize = 0;
};

static bool parse_args(int argc, char** argv, CmdArgs& a) {
    for (int i = 1; i < argc; i++) {
        const char* k = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "[args] %s 缺参数\n", name); return nullptr; }
            return argv[++i];
        };
        if (!strcmp(k, "--frames")) { a.framesRaw = next(k); }
        else if (!strcmp(k, "--mv")) { a.mvDir = next(k); }
        else if (!strcmp(k, "--depth")) { a.depthDir = next(k); }
        else if (!strcmp(k, "--mask")) { a.maskDir = next(k); a.responsiveMask = true; }
        else if (!strcmp(k, "--out")) { a.outRaw = next(k); }
        else if (!strcmp(k, "--in-w")) { const char* v = next(k); if (!v) return false; a.inW = atoi(v); }
        else if (!strcmp(k, "--in-h")) { const char* v = next(k); if (!v) return false; a.inH = atoi(v); }
        else if (!strcmp(k, "--out-w")) { const char* v = next(k); if (!v) return false; a.outW = atoi(v); }
        else if (!strcmp(k, "--out-h")) { const char* v = next(k); if (!v) return false; a.outH = atoi(v); }
        else if (!strcmp(k, "--frames-count")) { const char* v = next(k); if (!v) return false; a.framesCount = atoi(v); }
        else if (!strcmp(k, "--quality")) { const char* v = next(k); if (!v) return false; a.quality = 100 + atoi(v); }
        else if (!strcmp(k, "--device")) { const char* v = next(k); if (!v) return false; a.device = atoi(v); }
        else if (!strcmp(k, "--verbose")) { a.verbose = true; }
        else if (!strcmp(k, "--stream")) { a.stream = true; }
        else if (!strcmp(k, "--shm-name")) { a.shmName = next(k); a.stream = true; }
        else if (!strcmp(k, "--shm-slots")) {
            const char* v = next(k); if (!v) return false; a.shmSlots = (uint32_t)strtoul(v, nullptr, 10);
        }
        else if (!strcmp(k, "--shm-slot-size")) {
            const char* v = next(k); if (!v) return false; a.shmSlotSize = (uint32_t)strtoul(v, nullptr, 10);
        }
        else if (!strcmp(k, "--responsive-max")) {
            const char* v = next(k); if (!v) return false; a.responsiveMax = (float)atof(v);
        }
        else if (!strcmp(k, "--mv-path")) {
            const char* v = next(k);
            if (!v) return false;
            if (!strcmp(v, "highres")) a.lowResMv = false;
            else if (!strcmp(v, "lowres-depth")) a.lowResMv = true;
            else { fprintf(stderr, "[args] --mv-path must be highres or lowres-depth\n"); return false; }
        }
        else if (!strcmp(k, "--mv-upsample")) {
            const char* v = next(k);
            if (!v) return false;
            if (!strcmp(v, "nearest")) a.nearestMv = true;
            else if (!strcmp(v, "bilinear")) a.nearestMv = false;
            else { fprintf(stderr, "[args] --mv-upsample must be nearest or bilinear\n"); return false; }
        }
        else if (!strcmp(k, "--reset-frames")) { a.resetFrames = next(k); }
        else if (!strcmp(k, "--dump")) { a.dumpDir = next(k); }
        else if (!strcmp(k, "--dbg-upload")) { a.dbgUpload = next(k); }
        else { fprintf(stderr, "[args] 未知参数 %s\n", k); return false; }
        if (!k) return false;
    }
    const bool dimensions = a.inW > 0 && a.inH > 0 && a.outW > 0 && a.outH > 0 && a.framesCount > 0;
    const bool io = a.stream || (a.framesRaw && a.mvDir && a.outRaw);
    const bool depthOk = !a.lowResMv || a.stream || a.depthDir;
    const bool shmOk = !a.shmName || (a.shmSlots >= 2 && a.shmSlotSize >= sizeof(StreamHeader));
    return dimensions && io && depthOk && shmOk &&
           a.responsiveMax >= 0.0f && a.responsiveMax <= 1.0f;
}

struct D3D12Ctx {
    ID3D12Device* dev = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE fenceEvent = nullptr;
    uint64_t fenceVal = 0;

    ID3D12Resource* colorTex = nullptr;   // RGBA8 输入（inW x inH）
    ID3D12Resource* velTex = nullptr;     // R16G16_FLOAT motion vectors
    ID3D12Resource* depthTex = nullptr;   // optional R32_FLOAT inverse depth
    ID3D12Resource* maskTex = nullptr;    // optional R8_UNORM responsive mask
    ID3D12Resource* outTex = nullptr;     // RGBA8 输出（outW x outH）
    ID3D12Resource* uploadBuf = nullptr;
    ID3D12Resource* readbackBuf = nullptr;
    ID3D12DescriptorHeap* heap = nullptr;
    UINT heapInc = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE hColorCpu, hVelCpu, hOutCpu;
    uint32_t colorPitch = 0, velPitch = 0, depthPitch = 0, maskPitch = 0, readPitch = 0;
    uint64_t velOffset = 0, depthOffset = 0, maskOffset = 0;
    ID3D12Resource* dbgReadback = nullptr;  // 调试：上传后的颜色纹理回读

    ~D3D12Ctx() {
        if (dbgReadback) dbgReadback->Release();
        if (readbackBuf) readbackBuf->Release();
        if (uploadBuf) uploadBuf->Release();
        if (outTex) outTex->Release();
        if (maskTex) maskTex->Release();
        if (depthTex) depthTex->Release();
        if (velTex) velTex->Release();
        if (colorTex) colorTex->Release();
        if (heap) heap->Release();
        if (alloc) alloc->Release();
        if (list) list->Release();
        if (queue) queue->Release();
        if (fence) fence->Release();
        if (fenceEvent) CloseHandle(fenceEvent);
        if (dev) dev->Release();
    }

    bool wait_gpu() {
        fenceVal++;
        queue->Signal(fence, fenceVal);
        if (fence->GetCompletedValue() < fenceVal) {
            fence->SetEventOnCompletion(fenceVal, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        return true;
    }
};

static IDXGIAdapter1* pick_adapter(int wantIndex) {
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return nullptr;
    IDXGIAdapter1* best = nullptr;
    SIZE_T bestMem = 0;
    for (UINT i = 0;; i++) {
        IDXGIAdapter1* ad = nullptr;
        if (factory->EnumAdapters1(i, &ad) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc;
        ad->GetDesc1(&desc);
        if (wantIndex == (int)i) {
            factory->Release();
            return ad;
        }
        if (desc.DedicatedVideoMemory > bestMem) {
            if (best) best->Release();
            best = ad;
            bestMem = desc.DedicatedVideoMemory;
        } else {
            ad->Release();
        }
    }
    factory->Release();
    return best;
}

static ID3D12Resource* create_tex(ID3D12Device* dev, UINT w, UINT h,
                                  DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags,
                                  D3D12_RESOURCE_STATES initState,
                                  const char* name) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = fmt;
    d.SampleDesc = {1, 0};
    d.Flags = flags;
    ID3D12Resource* r = nullptr;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    HRESULT hr = dev->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &d, initState, nullptr, IID_PPV_ARGS(&r));
    if (FAILED(hr)) { fprintf(stderr, "[d3d12] 创建纹理 %s 失败 hr=0x%08lX\n", name, hr); return nullptr; }
    return r;
}

static bool init_d3d(int wantAdapter, UINT inW, UINT inH, UINT outW, UINT outH,
                     bool lowResMv, bool responsiveMask, D3D12Ctx& c) {
    IDXGIAdapter1* ad = pick_adapter(wantAdapter);
    if (!ad) { fprintf(stderr, "[d3d12] 没有可用适配器\n"); return false; }
    DXGI_ADAPTER_DESC1 desc;
    ad->GetDesc1(&desc);
    char name[256];
    WideCharToMultiByte(CP_ACP, 0, desc.Description, -1, name, sizeof(name), nullptr, nullptr);
    fprintf(stderr, "[d3d12] adapter: %s (%llu MiB)\n", name,
            (unsigned long long)(desc.DedicatedVideoMemory / 1024 / 1024));
    if (FAILED(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&c.dev)))) {
        ad->Release();
        fprintf(stderr, "[d3d12] 创建设备失败\n");
        return false;
    }
    ad->Release();

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(c.dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&c.queue)))) return false;
    if (FAILED(c.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&c.alloc)))) return false;
    if (FAILED(c.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, c.alloc, nullptr,
                                        IID_PPV_ARGS(&c.list)))) return false;
    c.list->Close();
    if (FAILED(c.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c.fence)))) return false;
    c.fenceEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);

    c.colorTex = create_tex(c.dev, inW, inH, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE,
                            D3D12_RESOURCE_STATE_COPY_DEST, "xess_color");
    const UINT velocityW = lowResMv ? inW : outW;
    const UINT velocityH = lowResMv ? inH : outH;
    c.velTex = create_tex(c.dev, velocityW, velocityH, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_NONE,
                          D3D12_RESOURCE_STATE_COPY_DEST, "xess_velocity");
    if (lowResMv)
        c.depthTex = create_tex(c.dev, inW, inH, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE,
                                D3D12_RESOURCE_STATE_COPY_DEST, "xess_depth");
    if (responsiveMask)
        c.maskTex = create_tex(c.dev, inW, inH, DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_COPY_DEST, "xess_responsive_mask");
    c.outTex = create_tex(c.dev, outW, outH, DXGI_FORMAT_R8G8B8A8_UNORM,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_COMMON, "xess_output");
    if (!c.colorTex || !c.velTex || !c.outTex || (lowResMv && !c.depthTex) ||
        (responsiveMask && !c.maskTex)) return false;

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 8;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(c.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&c.heap)))) return false;
    c.heapInc = c.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    c.hColorCpu = c.heap->GetCPUDescriptorHandleForHeapStart();

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels = 1;
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    c.dev->CreateShaderResourceView(c.colorTex, &srv, c.hColorCpu);
    c.hVelCpu = c.hColorCpu;
    c.hVelCpu.ptr += c.heapInc;
    srv.Format = DXGI_FORMAT_R16G16_FLOAT;
    c.dev->CreateShaderResourceView(c.velTex, &srv, c.hVelCpu);
    c.hOutCpu = c.hVelCpu;
    c.hOutCpu.ptr += c.heapInc;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    c.dev->CreateUnorderedAccessView(c.outTex, nullptr, &uav, c.hOutCpu);

    c.colorPitch = align_up(inW * 4, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    c.velPitch = align_up(velocityW * 4, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    c.depthPitch = align_up(inW * 4, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    c.maskPitch = align_up(inW, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    c.velOffset = (UINT64)c.colorPitch * inH;
    c.depthOffset = align_up((uint32_t)(c.velOffset + (UINT64)c.velPitch * velocityH),
                             D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    c.maskOffset = align_up((uint32_t)(c.depthOffset + (lowResMv ? (UINT64)c.depthPitch * inH : 0)),
                            D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    UINT64 upSize = c.maskOffset + (responsiveMask ? (UINT64)c.maskPitch * inH : 0);
    if (!responsiveMask) upSize = c.depthOffset + (lowResMv ? (UINT64)c.depthPitch * inH : 0);
    if (!lowResMv && !responsiveMask) upSize = c.velOffset + (UINT64)c.velPitch * velocityH;
    D3D12_RESOURCE_DESC ud{};
    ud.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    ud.Alignment = 0;
    ud.Width = upSize;
    ud.Height = 1;
    ud.DepthOrArraySize = 1;
    ud.MipLevels = 1;
    ud.SampleDesc = {1, 0};
    ud.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES upHp{};
    upHp.Type = D3D12_HEAP_TYPE_UPLOAD;
    if (FAILED(c.dev->CreateCommittedResource(&upHp, D3D12_HEAP_FLAG_NONE, &ud,
                                              D3D12_RESOURCE_STATE_GENERIC_READ,
                                              nullptr, IID_PPV_ARGS(&c.uploadBuf))))
        return false;
    c.readPitch = align_up(outW * 4, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Alignment = 0;
    rd.Width = (UINT64)c.readPitch * outH;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc = {1, 0};
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES rbHp{};
    rbHp.Type = D3D12_HEAP_TYPE_READBACK;
    if (FAILED(c.dev->CreateCommittedResource(&rbHp, D3D12_HEAP_FLAG_NONE, &rd,
                                              D3D12_RESOURCE_STATE_COPY_DEST,
                                              nullptr, IID_PPV_ARGS(&c.readbackBuf))))
        return false;
    // 调试回读缓冲（颜色纹理验证用，尺寸 = 输入）
    D3D12_RESOURCE_DESC drd{};
    drd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    drd.Alignment = 0;
    drd.Width = (UINT64)c.colorPitch * inH;
    drd.Height = 1;
    drd.DepthOrArraySize = 1;
    drd.MipLevels = 1;
    drd.SampleDesc = {1, 0};
    drd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(c.dev->CreateCommittedResource(&rbHp, D3D12_HEAP_FLAG_NONE, &drd,
                                              D3D12_RESOURCE_STATE_COPY_DEST,
                                              nullptr, IID_PPV_ARGS(&c.dbgReadback))))
        return false;
    return true;
}

// 上传一帧 + 执行 XeSS + 回读一帧（全部录制到同一个命令列表，一次提交）
static bool process_frame(D3D12Ctx& c, const uint8_t* rgb, const float* velLow,
                           const float* depthLow, const uint8_t* maskLow,
                           xess_context_handle_t xessCtx, const CmdArgs& a, FILE* outFp,
                           int frameIdx, bool resetHistory) {
    ID3D12GraphicsCommandList* L = c.list;
    c.alloc->Reset();
    L->Reset(c.alloc, nullptr);

    // 填充上行缓冲：颜色 RGB->RGBA；运动矢量 低分辨率 float32 -> 放大+半精度
    const int velocityW = a.lowResMv ? a.inW : a.outW;
    const int velocityH = a.lowResMv ? a.inH : a.outH;
    std::vector<uint16_t> velocity((size_t)velocityW * velocityH * 2);
    if (a.lowResMv) {
        for (size_t i = 0; i < (size_t)a.inW * a.inH * 2; ++i)
            velocity[i] = f32_to_f16(velLow[i]);
    } else {
        upsample_velocity(velLow, velocity.data(), a.inW, a.inH, a.outW, a.outH,
                          a.nearestMv);
    }

    void* upPtr = nullptr;
    if (FAILED(c.uploadBuf->Map(0, nullptr, &upPtr))) return false;
    uint8_t* colorDst = (uint8_t*)upPtr;
    for (int y = 0; y < a.inH; y++) {
        const uint8_t* src = rgb + (size_t)y * a.inW * 3;
        uint8_t* dst = colorDst + (size_t)y * c.colorPitch;
        for (int x = 0; x < a.inW; x++) {
            dst[x * 4 + 0] = src[x * 3 + 0];
            dst[x * 4 + 1] = src[x * 3 + 1];
            dst[x * 4 + 2] = src[x * 3 + 2];
            dst[x * 4 + 3] = 255;
        }
    }
    uint8_t* velDst = (uint8_t*)upPtr + c.velOffset;
    for (int y = 0; y < velocityH; y++)
        memcpy(velDst + (size_t)y * c.velPitch,
               (const uint8_t*)velocity.data() + (size_t)y * velocityW * 4, (size_t)velocityW * 4);
    if (a.lowResMv) {
        if (!depthLow) { c.uploadBuf->Unmap(0, nullptr); return false; }
        uint8_t* depthDst = (uint8_t*)upPtr + c.depthOffset;
        for (int y = 0; y < a.inH; ++y)
            memcpy(depthDst + (size_t)y * c.depthPitch,
                   depthLow + (size_t)y * a.inW, (size_t)a.inW * sizeof(float));
    }
    if (a.responsiveMask) {
        if (!maskLow) { c.uploadBuf->Unmap(0, nullptr); return false; }
        uint8_t* maskDst = (uint8_t*)upPtr + c.maskOffset;
        for (int y = 0; y < a.inH; ++y)
            memcpy(maskDst + (size_t)y * c.maskPitch,
                   maskLow + (size_t)y * a.inW, (size_t)a.inW);
    }
    c.uploadBuf->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dstC = { c.colorTex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0 };
    D3D12_TEXTURE_COPY_LOCATION srcC = { c.uploadBuf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, 0 };
    srcC.PlacedFootprint.Footprint = { DXGI_FORMAT_R8G8B8A8_UNORM, (UINT)a.inW, (UINT)a.inH, 1, c.colorPitch };
    L->CopyTextureRegion(&dstC, 0, 0, 0, &srcC, nullptr);
    D3D12_TEXTURE_COPY_LOCATION dstV = { c.velTex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0 };
    D3D12_TEXTURE_COPY_LOCATION srcV = { c.uploadBuf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                         c.velOffset };
    srcV.PlacedFootprint.Footprint = { DXGI_FORMAT_R16G16_FLOAT, (UINT)velocityW, (UINT)velocityH, 1, c.velPitch };
    L->CopyTextureRegion(&dstV, 0, 0, 0, &srcV, nullptr);
    if (a.lowResMv) {
        D3D12_TEXTURE_COPY_LOCATION dstD = { c.depthTex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0 };
        D3D12_TEXTURE_COPY_LOCATION srcD = { c.uploadBuf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                             c.depthOffset };
        srcD.PlacedFootprint.Footprint = { DXGI_FORMAT_R32_FLOAT, (UINT)a.inW, (UINT)a.inH, 1, c.depthPitch };
        L->CopyTextureRegion(&dstD, 0, 0, 0, &srcD, nullptr);
    }
    if (a.responsiveMask) {
        D3D12_TEXTURE_COPY_LOCATION dstM = { c.maskTex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0 };
        D3D12_TEXTURE_COPY_LOCATION srcM = { c.uploadBuf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                             c.maskOffset };
        srcM.PlacedFootprint.Footprint = { DXGI_FORMAT_R8_UNORM, (UINT)a.inW, (UINT)a.inH, 1, c.maskPitch };
        L->CopyTextureRegion(&dstM, 0, 0, 0, &srcM, nullptr);
    }

    auto bar = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = r;
        b.Transition.Subresource = 0;
        b.Transition.StateBefore = before;
        b.Transition.StateAfter = after;
        L->ResourceBarrier(1, &b);
    };
    bar(c.colorTex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    bar(c.velTex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (a.lowResMv)
        bar(c.depthTex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (a.responsiveMask)
        bar(c.maskTex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    bar(c.outTex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    xess_d3d12_execute_params_t exec{};
    exec.inputWidth = (uint32_t)a.inW;
    exec.inputHeight = (uint32_t)a.inH;
    exec.jitterOffsetX = 0.0f;
    exec.jitterOffsetY = 0.0f;
    exec.exposureScale = 1.0f;
    exec.resetHistory = resetHistory ? 1 : 0;
    exec.pColorTexture = c.colorTex;
    exec.pVelocityTexture = c.velTex;
    exec.pOutputTexture = c.outTex;
    exec.pDepthTexture = a.lowResMv ? c.depthTex : nullptr;
    exec.pExposureScaleTexture = nullptr;
    exec.pResponsivePixelMaskTexture = a.responsiveMask ? c.maskTex : nullptr;
    exec.pDescriptorHeap = nullptr;
    exec.descriptorHeapOffset = 0;
    int64_t rc = xessD3D12Execute(xessCtx, L, &exec);
    if (rc != XESS_RESULT_SUCCESS) {
        fprintf(stderr, "[xess] Execute 帧 %d 失败: %s\n", frameIdx, xess_err_str(rc));
        return false;
    }

    bar(c.outTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION srcO = { c.outTex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0 };
    D3D12_TEXTURE_COPY_LOCATION dstO = { c.readbackBuf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, 0 };
    dstO.PlacedFootprint.Footprint = { DXGI_FORMAT_R8G8B8A8_UNORM, (UINT)a.outW, (UINT)a.outH, 1, c.readPitch };
    L->CopyTextureRegion(&dstO, 0, 0, 0, &srcO, nullptr);
    bar(c.colorTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    bar(c.velTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (a.lowResMv)
        bar(c.depthTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (a.responsiveMask)
        bar(c.maskTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    bar(c.outTex, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);

    L->Close();
    ID3D12CommandList* lists[] = { L };
    c.queue->ExecuteCommandLists(1, lists);
    c.wait_gpu();

    void* rbPtr = nullptr;
    if (FAILED(c.readbackBuf->Map(0, nullptr, &rbPtr))) return false;
    const uint8_t* src = (const uint8_t*)rbPtr;
    std::vector<uint8_t> rowBuf((size_t)a.outW * 3);
    for (int y = 0; y < a.outH; y++) {
        const uint8_t* row = src + (size_t)y * c.readPitch;
        for (int x = 0; x < a.outW; x++) {          // RGBA -> RGB，跳过 alpha
            rowBuf[x * 3 + 0] = row[x * 4 + 0];
            rowBuf[x * 3 + 1] = row[x * 4 + 1];
            rowBuf[x * 3 + 2] = row[x * 4 + 2];
        }
        if (fwrite(rowBuf.data(), 3, (size_t)a.outW, outFp) != (size_t)a.outW) {
            c.readbackBuf->Unmap(0, nullptr);
            return false;
        }
    }
    c.readbackBuf->Unmap(0, nullptr);
    return true;
}

// 调试：只验证上传链路（颜色 RGB->RGBA -> CopyTextureRegion -> 回读），不跑 XeSS
static bool dbg_upload_frame(D3D12Ctx& c, const uint8_t* rgb, const CmdArgs& a,
                             const char* outPath) {
    ID3D12GraphicsCommandList* L = c.list;
    c.alloc->Reset();
    L->Reset(c.alloc, nullptr);
    void* upPtr = nullptr;
    if (FAILED(c.uploadBuf->Map(0, nullptr, &upPtr))) return false;
    uint8_t* colorDst = (uint8_t*)upPtr;
    for (int y = 0; y < a.inH; y++) {
        const uint8_t* src = rgb + (size_t)y * a.inW * 3;
        uint8_t* dst = colorDst + (size_t)y * c.colorPitch;
        for (int x = 0; x < a.inW; x++) {
            dst[x * 4 + 0] = src[x * 3 + 0];
            dst[x * 4 + 1] = src[x * 3 + 1];
            dst[x * 4 + 2] = src[x * 3 + 2];
            dst[x * 4 + 3] = 255;
        }
    }
    c.uploadBuf->Unmap(0, nullptr);
    D3D12_TEXTURE_COPY_LOCATION dstC = { c.colorTex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0 };
    D3D12_TEXTURE_COPY_LOCATION srcC = { c.uploadBuf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, 0 };
    srcC.PlacedFootprint.Footprint = { DXGI_FORMAT_R8G8B8A8_UNORM, (UINT)a.inW, (UINT)a.inH, 1, c.colorPitch };
    L->CopyTextureRegion(&dstC, 0, 0, 0, &srcC, nullptr);
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = c.colorTex;
    b.Transition.Subresource = 0;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    L->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION dstR = { c.dbgReadback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, 0 };
    dstR.PlacedFootprint.Footprint = { DXGI_FORMAT_R8G8B8A8_UNORM, (UINT)a.inW, (UINT)a.inH, 1, c.colorPitch };
    D3D12_TEXTURE_COPY_LOCATION srcT = { c.colorTex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0 };
    L->CopyTextureRegion(&dstR, 0, 0, 0, &srcT, nullptr);
    L->Close();
    ID3D12CommandList* lists[] = { L };
    c.queue->ExecuteCommandLists(1, lists);
    c.wait_gpu();
    void* rbPtr = nullptr;
    if (FAILED(c.dbgReadback->Map(0, nullptr, &rbPtr))) return false;
    FILE* fp = fopen(outPath, "wb");
    if (!fp) { c.dbgReadback->Unmap(0, nullptr); return false; }
    const uint8_t* src = (const uint8_t*)rbPtr;
    std::vector<uint8_t> rowBuf((size_t)a.inW * 3);
    for (int y = 0; y < a.inH; y++) {
        const uint8_t* row = src + (size_t)y * c.colorPitch;
        for (int x = 0; x < a.inW; x++) {           // RGBA -> RGB，跳过 alpha
            rowBuf[x * 3 + 0] = row[x * 4 + 0];
            rowBuf[x * 3 + 1] = row[x * 4 + 1];
            rowBuf[x * 3 + 2] = row[x * 4 + 2];
        }
        fwrite(rowBuf.data(), 3, (size_t)a.inW, fp);
    }
    fclose(fp);
    c.dbgReadback->Unmap(0, nullptr);
    return true;
}

struct StreamFrame {
    std::vector<uint8_t> color;
    std::vector<float> motion;
    std::vector<float> depth;
    std::vector<uint8_t> mask;
    uint32_t flags = 0;
};

// Returns 1 for a frame, 0 for EOS, and -1 for invalid/truncated input.
static int read_stream_frame(FILE* input, XessSharedRingReader* ring,
                             const CmdArgs& a, int expectedIndex, StreamFrame& frame) {
    std::vector<uint8_t> packet;
    StreamHeader header{};
    if (ring) {
        if (!ring->read(packet) || packet.size() < sizeof(header)) {
            fprintf(stderr, "[shm] missing packet at frame %d\n", expectedIndex);
            return -1;
        }
        memcpy(&header, packet.data(), sizeof(header));
    } else if (!read_exact(input, &header, sizeof(header))) {
        fprintf(stderr, "[stream] truncated header at frame %d\n", expectedIndex);
        return -1;
    }
    if (memcmp(header.magic, "XSPK", 4) || header.version != 1 ||
        header.headerSize != sizeof(StreamHeader)) {
        fprintf(stderr, "[stream] invalid protocol header at frame %d\n", expectedIndex);
        return -1;
    }
    if (header.flags & STREAM_FLAG_EOS) {
        if (header.frameIndex != (uint32_t)expectedIndex || header.colorBytes ||
            header.motionBytes || header.depthBytes || header.maskBytes) {
            fprintf(stderr, "[stream] malformed EOS packet\n");
            return -1;
        }
        return 0;
    }
    const uint64_t total64 = (uint64_t)header.colorBytes + header.motionBytes +
                             header.depthBytes + header.maskBytes;
    if (total64 > 512ull * 1024 * 1024 || header.frameIndex != (uint32_t)expectedIndex ||
        header.width != (uint32_t)a.inW || header.height != (uint32_t)a.inH ||
        header.pixelFormat != 1) {
        fprintf(stderr, "[stream] metadata mismatch at frame %d\n", expectedIndex);
        return -1;
    }
    const size_t pixels = (size_t)a.inW * a.inH;
    if (header.colorBytes != pixels * 3 || header.motionBytes != pixels * 2 * sizeof(float) ||
        (header.depthBytes && header.depthBytes != pixels * sizeof(float)) ||
        (header.maskBytes && header.maskBytes != pixels) ||
        (a.lowResMv && header.depthBytes != pixels * sizeof(float)) ||
        (a.responsiveMask && header.maskBytes != pixels)) {
        fprintf(stderr, "[stream] payload sizes mismatch at frame %d\n", expectedIndex);
        return -1;
    }
    std::vector<uint8_t> payload((size_t)total64);
    if (ring) {
        if (packet.size() != sizeof(header) + payload.size()) {
            fprintf(stderr, "[shm] packet length mismatch at frame %d\n", expectedIndex);
            return -1;
        }
        memcpy(payload.data(), packet.data() + sizeof(header), payload.size());
    } else if (!read_exact(input, payload.data(), payload.size())) {
        fprintf(stderr, "[stream] truncated payload at frame %d\n", expectedIndex);
        return -1;
    }
    if (crc32_bytes(payload.data(), payload.size()) != header.checksum) {
        fprintf(stderr, "[stream] checksum mismatch at frame %d\n", expectedIndex);
        return -1;
    }
    size_t offset = 0;
    frame.color.assign(payload.begin(), payload.begin() + header.colorBytes);
    offset += header.colorBytes;
    frame.motion.resize(pixels * 2);
    memcpy(frame.motion.data(), payload.data() + offset, header.motionBytes);
    offset += header.motionBytes;
    frame.depth.clear();
    if (header.depthBytes) {
        frame.depth.resize(pixels);
        memcpy(frame.depth.data(), payload.data() + offset, header.depthBytes);
    }
    offset += header.depthBytes;
    frame.mask.clear();
    if (header.maskBytes)
        frame.mask.assign(payload.begin() + offset, payload.begin() + offset + header.maskBytes);
    frame.flags = header.flags;
    return 1;
}

int main(int argc, char** argv) {
    CmdArgs a;
    if (!parse_args(argc, argv, a)) {
        fprintf(stderr, "usage: xess-vsr.exe [--stream [--shm-name NAME --shm-slots N --shm-slot-size N] "
                        "| --frames in.raw --mv DIR --out out.raw] "
                        "--in-w W --in-h H --out-w OW --out-h OH --frames-count N "
                        "[--mv-path highres|lowres-depth] [--depth DIR] [--mask DIR]\n");
        return 2;
    }

    if (a.stream) {
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
    }

    xess_version_t ver{};
    if (xessGetVersion(&ver) != XESS_RESULT_SUCCESS) {
        fprintf(stderr, "[xess] xessGetVersion 失败（libxess.dll 未找到或损坏）\n");
        return 1;
    }
    fprintf(stderr, "[xess] libxess.dll %u.%u.%u\n", ver.major, ver.minor, ver.patch);

    D3D12Ctx d3d;
    if (!init_d3d(a.device, a.inW, a.inH, a.outW, a.outH,
                  a.lowResMv, a.responsiveMask, d3d)) return 1;

    if (a.dbgUpload) {
        std::vector<uint8_t> rgb((size_t)a.inW * a.inH * 3);
        FILE* rawIn = fopen(a.framesRaw, "rb");
        if (!rawIn) { fprintf(stderr, "[io] 打不开 %s\n", a.framesRaw); return 1; }
        fread(rgb.data(), 3, (size_t)a.inW * a.inH, rawIn);
        fclose(rawIn);
        if (!dbg_upload_frame(d3d, rgb.data(), a, a.dbgUpload)) return 1;
        fprintf(stderr, "[dbg] upload readback complete -> %s\n", a.dbgUpload);
        return 0;
    }

    // ---- XeSS 上下文 + 初始化 ----
    xess_context_handle_t ctx = nullptr;
    int64_t rc = xessD3D12CreateContext(d3d.dev, &ctx);
    if (rc != XESS_RESULT_SUCCESS || !ctx) {
        fprintf(stderr, "[xess] 创建上下文失败: %s\n", xess_err_str(rc));
        return 1;
    }
    xess_2d_t inRes{};
    rc = xessGetInputResolution(ctx, &xess_2d_t{(uint32_t)a.outW, (uint32_t)a.outH},
                                (xess_quality_settings_t)a.quality, &inRes);
    if (rc == XESS_RESULT_SUCCESS)
        fprintf(stderr, "[xess] expected input %ux%u, actual %ux%u\n",
                inRes.x, inRes.y, a.inW, a.inH);
    else
        fprintf(stderr, "[xess] xessGetInputResolution: %s\n", xess_err_str(rc));

    xess_d3d12_init_params_t params{};
    params.outputResolution = { (uint32_t)a.outW, (uint32_t)a.outH };
    params.qualitySetting = (xess_quality_settings_t)a.quality;
    params.initFlags = XESS_INIT_FLAG_LDR_INPUT_COLOR;
    if (a.lowResMv) params.initFlags |= XESS_INIT_FLAG_INVERTED_DEPTH;
    else params.initFlags |= XESS_INIT_FLAG_HIGH_RES_MV;
    if (a.responsiveMask) params.initFlags |= XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK;
    params.creationNodeMask = 0;
    params.visibleNodeMask = 0;
    params.pTempBufferHeap = nullptr;
    params.bufferHeapOffset = 0;
    params.pTempTextureHeap = nullptr;
    params.textureHeapOffset = 0;
    params.pPipelineLibrary = nullptr;

    rc = xessD3D12BuildPipelines(ctx, nullptr, true, params.initFlags);
    if (rc != XESS_RESULT_SUCCESS)
        fprintf(stderr, "[xess] BuildPipelines: %s (继续)\n", xess_err_str(rc));
    rc = xessD3D12Init(ctx, &params);
    if (rc != XESS_RESULT_SUCCESS) {
        fprintf(stderr, "[xess] Init 失败: %s\n", xess_err_str(rc));
        xessDestroyContext(ctx);
        return 1;
    }
    rc = xessSetVelocityScale(ctx, 1.0f, 1.0f);
    if (rc != XESS_RESULT_SUCCESS)
        fprintf(stderr, "[xess] SetVelocityScale: %s\n", xess_err_str(rc));
    if (a.responsiveMask) {
        rc = xessSetMaxResponsiveMaskValue(ctx, a.responsiveMax);
        if (rc != XESS_RESULT_SUCCESS)
            fprintf(stderr, "[xess] SetMaxResponsiveMaskValue: %s\n", xess_err_str(rc));
    }
    fprintf(stderr, "[xess] init complete, mv=%s, depth=%s, mask=%s\n",
            a.lowResMv ? "lowres" : "highres", a.lowResMv ? "inverse" : "off",
            a.responsiveMask ? "on" : "off");

    if (a.dumpDir) {
        xess_dump_parameters_t dump{};
        dump.path = a.dumpDir;
        dump.frame_idx = 0;
        dump.frame_count = 1;
        dump.dump_elements_mask = XESS_DUMP_INPUT_COLOR | XESS_DUMP_INPUT_VELOCITY |
                                  XESS_DUMP_OUTPUT | XESS_DUMP_EXECUTION_PARAMETERS;
        rc = xessStartDump(ctx, &dump);
        fprintf(stderr, "[xess] dump: %s\n", xess_err_str(rc));
    }

    FILE* rawIn = a.stream ? stdin : fopen(a.framesRaw, "rb");
    if (!rawIn) { fprintf(stderr, "[io] cannot open %s\n", a.framesRaw); xessDestroyContext(ctx); return 1; }
    FILE* outFp = a.stream ? stdout : fopen(a.outRaw, "wb");
    if (!outFp) {
        fprintf(stderr, "[io] cannot open %s\n", a.outRaw);
        if (!a.stream) fclose(rawIn);
        xessDestroyContext(ctx);
        return 1;
    }
    XessSharedRingReader sharedRing;
    XessSharedRingReader* sharedRingPtr = nullptr;
    if (a.shmName) {
        if (!sharedRing.open(a.shmName, a.shmSlots, a.shmSlotSize)) {
            if (!a.stream) { fclose(outFp); fclose(rawIn); }
            xessDestroyContext(ctx);
            return 1;
        }
        sharedRingPtr = &sharedRing;
    }

    std::vector<uint8_t> rgb((size_t)a.inW * a.inH * 3);
    std::vector<float> velLow((size_t)a.inW * a.inH * 2);
    std::vector<float> depthLow(a.lowResMv ? (size_t)a.inW * a.inH : 0);
    std::vector<uint8_t> maskLow(a.responsiveMask ? (size_t)a.inW * a.inH : 0);
    std::vector<uint8_t> resetHistory((size_t)a.framesCount, 0);
    resetHistory[0] = 1;
    if (!a.stream && a.resetFrames) {
        FILE* resetFile = fopen(a.resetFrames, "r");
        if (!resetFile) {
            fprintf(stderr, "[io] cannot open reset-frame file %s\n", a.resetFrames);
            if (!a.stream) { fclose(outFp); fclose(rawIn); }
            xessDestroyContext(ctx);
            return 1;
        }
        int resetIndex = 0;
        while (fscanf(resetFile, "%d", &resetIndex) == 1) {
            if (resetIndex >= 0 && resetIndex < a.framesCount)
                resetHistory[(size_t)resetIndex] = 1;
        }
        fclose(resetFile);
    }
    char path[1024];
    bool complete = true;
    StreamFrame streamFrame;

    for (int i = 0; i < a.framesCount; i++) {
        bool reset = resetHistory[(size_t)i] != 0;
        if (a.stream) {
            const int streamResult = read_stream_frame(rawIn, sharedRingPtr, a, i, streamFrame);
            if (streamResult != 1) {
                fprintf(stderr, "[stream] expected frame %d, got %s\n", i,
                        streamResult == 0 ? "early EOS" : "invalid input");
                complete = false;
                break;
            }
            rgb = streamFrame.color;
            velLow = streamFrame.motion;
            if (a.lowResMv) depthLow = streamFrame.depth;
            if (a.responsiveMask) maskLow = streamFrame.mask;
            reset = (streamFrame.flags & (STREAM_FLAG_RESET | STREAM_FLAG_SCENE_CUT)) != 0;
        } else {
            if (!read_exact(rawIn, rgb.data(), rgb.size())) {
                fprintf(stderr, "[io] incomplete color frame %d\n", i);
                complete = false;
                break;
            }
            snprintf(path, sizeof(path), "%s\\mv_%06d.bin", a.mvDir, i);
            FILE* file = fopen(path, "rb");
            if (!file || !read_exact(file, velLow.data(), velLow.size() * sizeof(float))) {
                fprintf(stderr, "[io] missing/incomplete motion file %s\n", path);
                if (file) fclose(file);
                complete = false;
                break;
            }
            fclose(file);
            if (a.lowResMv) {
                snprintf(path, sizeof(path), "%s\\depth_%06d.bin", a.depthDir, i);
                file = fopen(path, "rb");
                if (!file || !read_exact(file, depthLow.data(), depthLow.size() * sizeof(float))) {
                    fprintf(stderr, "[io] missing/incomplete depth file %s\n", path);
                    if (file) fclose(file);
                    complete = false;
                    break;
                }
                fclose(file);
            }
            if (a.responsiveMask) {
                snprintf(path, sizeof(path), "%s\\mask_%06d.bin", a.maskDir, i);
                file = fopen(path, "rb");
                if (!file || !read_exact(file, maskLow.data(), maskLow.size())) {
                    fprintf(stderr, "[io] missing/incomplete mask file %s\n", path);
                    if (file) fclose(file);
                    complete = false;
                    break;
                }
                fclose(file);
            }
        }

        if (!process_frame(d3d, rgb.data(), velLow.data(),
                           a.lowResMv ? depthLow.data() : nullptr,
                           a.responsiveMask ? maskLow.data() : nullptr,
                           ctx, a, outFp, i, reset)) {
            complete = false;
            break;
        }
        if (a.verbose || (i + 1) % 25 == 0)
            fprintf(stderr, "[xess] %d/%d\n", i + 1, a.framesCount);
    }

    if (complete && a.stream) {
        const int eosResult = read_stream_frame(rawIn, sharedRingPtr, a, a.framesCount, streamFrame);
        if (eosResult != 0) {
            fprintf(stderr, "[stream] missing EOS or extra frame\n");
            complete = false;
        }
        fflush(outFp);
    }

    if (!a.stream) { fclose(outFp); fclose(rawIn); }
    xessDestroyContext(ctx);
    fprintf(stderr, "[xess] %s\n", complete ? "complete" : "failed");
    return complete ? 0 : 1;
}
