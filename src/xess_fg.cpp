// xess-fg: feed decoded video frames, motion vectors, and depth to the
// documented XeSS-FG D3D12 swap-chain API.
//
// Input:
//   rgb24 raw frames + one float32 [x,y] motion-vector file per frame.
// Output:
//   rgb24 raw generated frames (normally N-1 frames for N inputs).

// The swap-chain API is presentation-oriented and does not expose a generated
// texture directly. The default direct backend records the native swap chain
// created by XeSS-FG and reads its last presented back buffer. The legacy
// Windows Graphics Capture backend remains available for diagnostics.

// Usage:
//   xess-fg.exe --frames frames.raw --mv mvs --width 1280 --height 720
//               --frames-count 30 --fps 24 --out generated.raw
//               [--device N] [--verbose] [--dump-buffers DIR]


#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#include <d3d12.h>
#include <d3d11.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <fcntl.h>
#include <io.h>
#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

#include "xess_fg/xefg_swapchain.h"
#include "xess_fg/xefg_swapchain_d3d12.h"
#include "xess_fg/xefg_swapchain_debug.h"
#include "xell/xell.h"
#include "xell/xell_d3d12.h"
#include "shm_ring_win.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr UINT kBufferCount = 4;
constexpr uint32_t kStreamFlagReset = 1u << 0;
constexpr uint32_t kStreamFlagSceneCut = 1u << 1;
constexpr uint32_t kStreamFlagEos = 1u << 2;

#pragma pack(push, 1)
struct StreamHeader {
    char magic[4];
    uint16_t version;
    uint16_t header_size;
    uint32_t frame_index;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    uint32_t flags;
    uint32_t color_bytes;
    uint32_t motion_bytes;
    uint32_t depth_bytes;
    uint32_t mask_bytes;
    uint32_t checksum;
};
#pragma pack(pop)
static_assert(sizeof(StreamHeader) == 48, "stream header ABI mismatch");

uint32_t crc32_bytes(const uint8_t* data, size_t size) {
    static uint32_t table[256]{};
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t value = 0; value < 256; ++value) {
            uint32_t crc = value;
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc >> 1) ^ (0xedb88320u & static_cast<uint32_t>(-
                    static_cast<int32_t>(crc & 1)));
            table[value] = crc;
        }
        initialized = true;
    }
    uint32_t crc = 0xffffffffu;
    for (size_t index = 0; index < size; ++index)
        crc = table[(crc ^ data[index]) & 0xffu] ^ (crc >> 8);
    return ~crc;
}

bool read_exact(FILE* file, void* destination, size_t size) {
    auto* bytes = static_cast<uint8_t*>(destination);
    size_t done = 0;
    while (done < size) {
        size_t count = fread(bytes + done, 1, size - done, file);
        if (!count) return false;
        done += count;
    }
    return true;
}

uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

uint16_t f32_to_f16(float f) {
    uint32_t x = 0;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu)
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 0x0200u : 0));
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half & 1))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = static_cast<uint32_t>(exp) << 10;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (mantissa & 0x2000u))) {
        ++half;
        if (half == 0x7c00u) return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | half | (mantissa >> 13));
}

const char* fg_result_string(xefg_swapchain_result_t result) {
    switch (result) {
    case XEFG_SWAPCHAIN_RESULT_SUCCESS: return "SUCCESS";
    case XEFG_SWAPCHAIN_RESULT_WARNING_OLD_DRIVER: return "WARNING_OLD_DRIVER";
    case XEFG_SWAPCHAIN_RESULT_WARNING_TOO_FEW_FRAMES: return "WARNING_TOO_FEW_FRAMES";
    case XEFG_SWAPCHAIN_RESULT_WARNING_FRAMES_ID_MISMATCH: return "WARNING_FRAMES_ID_MISMATCH";
    case XEFG_SWAPCHAIN_RESULT_WARNING_MISSING_PRESENT_STATUS: return "WARNING_MISSING_PRESENT_STATUS";
    case XEFG_SWAPCHAIN_RESULT_WARNING_RESOURCE_SIZES_MISMATCH: return "WARNING_RESOURCE_SIZES_MISMATCH";
    case XEFG_SWAPCHAIN_RESULT_ERROR_UNSUPPORTED_DEVICE: return "UNSUPPORTED_DEVICE";
    case XEFG_SWAPCHAIN_RESULT_ERROR_UNSUPPORTED_DRIVER: return "UNSUPPORTED_DRIVER";
    case XEFG_SWAPCHAIN_RESULT_ERROR_UNINITIALIZED: return "UNINITIALIZED";
    case XEFG_SWAPCHAIN_RESULT_ERROR_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case XEFG_SWAPCHAIN_RESULT_ERROR_DEVICE_OUT_OF_MEMORY: return "DEVICE_OUT_OF_MEMORY";
    case XEFG_SWAPCHAIN_RESULT_ERROR_DEVICE: return "DEVICE";
    case XEFG_SWAPCHAIN_RESULT_ERROR_NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
    case XEFG_SWAPCHAIN_RESULT_ERROR_INVALID_CONTEXT: return "INVALID_CONTEXT";
    case XEFG_SWAPCHAIN_RESULT_ERROR_OPERATION_IN_PROGRESS: return "OPERATION_IN_PROGRESS";
    case XEFG_SWAPCHAIN_RESULT_ERROR_UNSUPPORTED: return "UNSUPPORTED";
    case XEFG_SWAPCHAIN_RESULT_ERROR_CANT_LOAD_LIBRARY: return "CANT_LOAD_LIBRARY";
    case XEFG_SWAPCHAIN_RESULT_ERROR_MISMATCH_INPUT_RESOURCES: return "MISMATCH_INPUT_RESOURCES";
    case XEFG_SWAPCHAIN_RESULT_ERROR_INCORRECT_OUTPUT_RESOURCES: return "INCORRECT_OUTPUT_RESOURCES";
    case XEFG_SWAPCHAIN_RESULT_ERROR_INCORRECT_INPUT_RESOURCES: return "INCORRECT_INPUT_RESOURCES";
    case XEFG_SWAPCHAIN_RESULT_ERROR_LATENCY_REDUCTION_UNSUPPORTED: return "LATENCY_REDUCTION_UNSUPPORTED";
    case XEFG_SWAPCHAIN_RESULT_ERROR_LATENCY_REDUCTION_FUNCTION_MISSING: return "LATENCY_REDUCTION_FUNCTION_MISSING";
    case XEFG_SWAPCHAIN_RESULT_ERROR_HRESULT_FAILURE: return "HRESULT_FAILURE";
    case XEFG_SWAPCHAIN_RESULT_ERROR_DXGI_INVALID_CALL: return "DXGI_INVALID_CALL";
    case XEFG_SWAPCHAIN_RESULT_ERROR_POINTER_STILL_IN_USE: return "POINTER_STILL_IN_USE";
    case XEFG_SWAPCHAIN_RESULT_ERROR_INVALID_DESCRIPTOR_HEAP: return "INVALID_DESCRIPTOR_HEAP";
    case XEFG_SWAPCHAIN_RESULT_ERROR_WRONG_CALL_ORDER: return "WRONG_CALL_ORDER";
    case XEFG_SWAPCHAIN_RESULT_ERROR_UNKNOWN: return "UNKNOWN";
    default: return "?";
    }
}

bool fg_ok(xefg_swapchain_result_t result, const char* operation) {
    if (result == XEFG_SWAPCHAIN_RESULT_SUCCESS) return true;
    fprintf(stderr, "[xefg] %s: %s (%d)\n", operation, fg_result_string(result), static_cast<int>(result));
    return result > 0;
}

bool hr_ok(HRESULT hr, const char* operation) {
    if (SUCCEEDED(hr)) return true;
    fprintf(stderr, "[d3d12] %s: HRESULT 0x%08lx\n", operation, static_cast<unsigned long>(hr));
    return false;
}

struct Args {
    const char* frames = nullptr;
    const char* mv_dir = nullptr;
    const char* depth_dir = nullptr;
    const char* output = nullptr;
    const char* dump_buffers = nullptr;
    const char* reset_frames = nullptr;
    const char* ui_mask_dir = nullptr;
    const char* shm_name = nullptr;
    uint32_t shm_slots = 0;
    uint32_t shm_slot_size = 0;
    int width = 0;
    int height = 0;
    int frame_count = 0;
    double fps = 24.0;
    int device = -1;
    bool verbose = false;
    bool allow_overlay = false;
    bool stream = false;
    bool direct_capture = true;
};

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "[args] %s requires a value\n", key);
                return nullptr;
            }
            return argv[++i];
        };
        if (!strcmp(key, "--frames")) args.frames = next();
        else if (!strcmp(key, "--mv")) args.mv_dir = next();
        else if (!strcmp(key, "--depth")) args.depth_dir = next();
        else if (!strcmp(key, "--out")) args.output = next();
        else if (!strcmp(key, "--width")) { const char* v = next(); if (!v) return false; args.width = atoi(v); }
        else if (!strcmp(key, "--height")) { const char* v = next(); if (!v) return false; args.height = atoi(v); }
        else if (!strcmp(key, "--frames-count")) { const char* v = next(); if (!v) return false; args.frame_count = atoi(v); }
        else if (!strcmp(key, "--fps")) { const char* v = next(); if (!v) return false; args.fps = atof(v); }
        else if (!strcmp(key, "--device")) { const char* v = next(); if (!v) return false; args.device = atoi(v); }
        else if (!strcmp(key, "--dump-buffers")) args.dump_buffers = next();
        else if (!strcmp(key, "--reset-frames")) args.reset_frames = next();
        else if (!strcmp(key, "--ui-mask")) args.ui_mask_dir = next();
        else if (!strcmp(key, "--verbose")) args.verbose = true;
        else if (!strcmp(key, "--allow-overlay")) args.allow_overlay = true;
        else if (!strcmp(key, "--capture-mode")) {
            const char* v = next();
            if (!v) return false;
            if (!strcmp(v, "direct")) args.direct_capture = true;
            else if (!strcmp(v, "window")) args.direct_capture = false;
            else {
                fprintf(stderr, "[args] --capture-mode must be direct or window\n");
                return false;
            }
        }
        else if (!strcmp(key, "--stream")) args.stream = true;
        else if (!strcmp(key, "--shm-name")) { args.shm_name = next(); args.stream = true; }
        else if (!strcmp(key, "--shm-slots")) {
            const char* v = next(); if (!v) return false; args.shm_slots = (uint32_t)strtoul(v, nullptr, 10);
        }
        else if (!strcmp(key, "--shm-slot-size")) {
            const char* v = next(); if (!v) return false; args.shm_slot_size = (uint32_t)strtoul(v, nullptr, 10);
        }
        else {
            fprintf(stderr, "[args] unknown option: %s\n", key);
            return false;
        }
    }
    const bool io = args.stream || (args.frames && args.mv_dir && args.output);
    const bool shm_ok = !args.shm_name ||
        (args.shm_slots >= 2 && args.shm_slot_size >= sizeof(StreamHeader));
    return io && shm_ok && args.width > 0 && args.height > 0 &&
           args.frame_count > 1 && args.fps > 0.0;
}

bool process_running(const wchar_t* executable_name) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return false;
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    bool found = false;
    if (Process32FirstW(snapshot, &entry)) {
        do {
            if (_wcsicmp(entry.szExeFile, executable_name) == 0) {
                found = true;
                break;
            }
        } while (Process32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);
    return found;
}

LRESULT CALLBACK window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_CLOSE) {
        DestroyWindow(hwnd);
        return 0;
    }
    return DefWindowProcW(hwnd, message, wparam, lparam);
}

HWND create_present_window(int width, int height) {
    HINSTANCE instance = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance;
    wc.lpszClassName = L"XeSSFGOfflineWindow";
    if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return nullptr;

    // Keep one compositor-visible column so DXGI does not classify the swap chain as occluded.
    // Windows Graphics Capture still captures the complete borderless window.
    const int x = std::max(0, GetSystemMetrics(SM_CXSCREEN) - 1);
    const int y = 0;
    HWND hwnd = CreateWindowExW(WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
                                wc.lpszClassName, L"XeSS-FG offline worker", WS_POPUP,
                                x, y, width, height, nullptr, nullptr, instance, nullptr);
    if (hwnd) {
        ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        SetWindowPos(hwnd, HWND_TOPMOST, x, y, width, height, SWP_NOACTIVATE | SWP_SHOWWINDOW);
    }
    return hwnd;
}

ComPtr<IDXGIAdapter1> pick_adapter(IDXGIFactory4* factory, int requested) {
    ComPtr<IDXGIAdapter1> best;
    SIZE_T best_memory = 0;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        HRESULT hr = factory->EnumAdapters1(index, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(hr)) continue;
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (requested == static_cast<int>(index)) return adapter;
        if (!best || desc.DedicatedVideoMemory > best_memory) {
            best = adapter;
            best_memory = desc.DedicatedVideoMemory;
        }
    }
    return requested < 0 ? best : nullptr;
}

// XeSS-FG's documented descriptor initialization creates a native DXGI swap
// chain through the factory supplied by the application, then returns only its
// proxy. This forwarding factory records the created native object without
// changing its reference count or ownership while initialization is in flight.
// Once initialization completes, the caller obtains its own reference through
// QueryInterface and can read the native presented buffers directly.
class RecordingFactory final : public IDXGIFactory2 {
public:
    explicit RecordingFactory(IDXGIFactory2* inner) : inner_(inner) {}

    IDXGISwapChain1* captured_swapchain() const { return captured_swapchain_; }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** object) override {
        if (!object) return E_POINTER;
        *object = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) ||
            riid == __uuidof(IDXGIFactory) || riid == __uuidof(IDXGIFactory1) ||
            riid == __uuidof(IDXGIFactory2)) {
            *object = static_cast<IDXGIFactory2*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() override {
        return static_cast<ULONG>(InterlockedIncrement(&references_));
    }

    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG value = static_cast<ULONG>(InterlockedDecrement(&references_));
        if (!value) delete this;
        return value;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID name, UINT size,
                                             const void* data) override {
        return inner_->SetPrivateData(name, size, data);
    }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID name,
                                                      const IUnknown* unknown) override {
        return inner_->SetPrivateDataInterface(name, unknown);
    }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID name, UINT* size, void* data) override {
        return inner_->GetPrivateData(name, size, data);
    }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID riid, void** parent) override {
        return inner_->GetParent(riid, parent);
    }
    HRESULT STDMETHODCALLTYPE EnumAdapters(UINT index, IDXGIAdapter** adapter) override {
        return inner_->EnumAdapters(index, adapter);
    }
    HRESULT STDMETHODCALLTYPE MakeWindowAssociation(HWND window, UINT flags) override {
        return inner_->MakeWindowAssociation(window, flags);
    }
    HRESULT STDMETHODCALLTYPE GetWindowAssociation(HWND* window) override {
        return inner_->GetWindowAssociation(window);
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChain(IUnknown* device, DXGI_SWAP_CHAIN_DESC* desc,
                                              IDXGISwapChain** swapchain) override {
        return inner_->CreateSwapChain(device, desc, swapchain);
    }
    HRESULT STDMETHODCALLTYPE CreateSoftwareAdapter(HMODULE module,
                                                    IDXGIAdapter** adapter) override {
        return inner_->CreateSoftwareAdapter(module, adapter);
    }
    HRESULT STDMETHODCALLTYPE EnumAdapters1(UINT index, IDXGIAdapter1** adapter) override {
        return inner_->EnumAdapters1(index, adapter);
    }
    BOOL STDMETHODCALLTYPE IsCurrent() override { return inner_->IsCurrent(); }
    BOOL STDMETHODCALLTYPE IsWindowedStereoEnabled() override {
        return inner_->IsWindowedStereoEnabled();
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForHwnd(
        IUnknown* device, HWND window, const DXGI_SWAP_CHAIN_DESC1* desc,
        const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen_desc,
        IDXGIOutput* restrict_to_output, IDXGISwapChain1** swapchain) override {
        HRESULT hr = inner_->CreateSwapChainForHwnd(
            device, window, desc, fullscreen_desc, restrict_to_output, swapchain);
        if (SUCCEEDED(hr) && swapchain && *swapchain) captured_swapchain_ = *swapchain;
        return hr;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForCoreWindow(
        IUnknown* device, IUnknown* window, const DXGI_SWAP_CHAIN_DESC1* desc,
        IDXGIOutput* restrict_to_output, IDXGISwapChain1** swapchain) override {
        return inner_->CreateSwapChainForCoreWindow(
            device, window, desc, restrict_to_output, swapchain);
    }
    HRESULT STDMETHODCALLTYPE GetSharedResourceAdapterLuid(HANDLE resource,
                                                           LUID* luid) override {
        return inner_->GetSharedResourceAdapterLuid(resource, luid);
    }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusWindow(HWND window, UINT message,
                                                         DWORD* cookie) override {
        return inner_->RegisterStereoStatusWindow(window, message, cookie);
    }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusEvent(HANDLE event, DWORD* cookie) override {
        return inner_->RegisterStereoStatusEvent(event, cookie);
    }
    void STDMETHODCALLTYPE UnregisterStereoStatus(DWORD cookie) override {
        inner_->UnregisterStereoStatus(cookie);
    }
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusWindow(HWND window, UINT message,
                                                            DWORD* cookie) override {
        return inner_->RegisterOcclusionStatusWindow(window, message, cookie);
    }
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusEvent(HANDLE event,
                                                           DWORD* cookie) override {
        return inner_->RegisterOcclusionStatusEvent(event, cookie);
    }
    void STDMETHODCALLTYPE UnregisterOcclusionStatus(DWORD cookie) override {
        inner_->UnregisterOcclusionStatus(cookie);
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForComposition(
        IUnknown* device, const DXGI_SWAP_CHAIN_DESC1* desc,
        IDXGIOutput* restrict_to_output, IDXGISwapChain1** swapchain) override {
        return inner_->CreateSwapChainForComposition(
            device, desc, restrict_to_output, swapchain);
    }

private:
    ~RecordingFactory() = default;
    volatile LONG references_ = 1;
    ComPtr<IDXGIFactory2> inner_;
    IDXGISwapChain1* captured_swapchain_ = nullptr; // Non-owning during init.
};

ComPtr<ID3D12Resource> create_texture(ID3D12Device* device, UINT width, UINT height,
                                      DXGI_FORMAT format, D3D12_RESOURCE_STATES initial_state,
                                      const wchar_t* name) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> resource;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, initial_state,
                                               nullptr, IID_PPV_ARGS(&resource)))) return nullptr;
    resource->SetName(name);
    return resource;
}

ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, uint64_t bytes,
                                     D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES state,
                                     const wchar_t* name) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = heap_type;
    ComPtr<ID3D12Resource> resource;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state,
                                               nullptr, IID_PPV_ARGS(&resource)))) return nullptr;
    resource->SetName(name);
    return resource;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &barrier);
}

struct Runtime {
    HWND hwnd = nullptr;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;

    xell_context_handle_t xell = nullptr;
    xefg_swapchain_handle_t xefg = nullptr;
    ComPtr<IDXGISwapChain4> swapchain;
    std::vector<ComPtr<ID3D12Resource>> backbuffers;
    ComPtr<IDXGISwapChain4> native_swapchain;
    std::vector<ComPtr<ID3D12Resource>> native_backbuffers;

    ComPtr<ID3D12Resource> velocity;
    ComPtr<ID3D12Resource> depth;
    ComPtr<ID3D12Resource> ui;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Resource> readback;
    uint32_t color_pitch = 0;
    uint32_t velocity_pitch = 0;
    uint32_t depth_pitch = 0;
    uint32_t ui_pitch = 0;
    uint32_t readback_pitch = 0;
    uint64_t velocity_offset = 0;
    uint64_t depth_offset = 0;
    uint64_t ui_offset = 0;

    ~Runtime() {
        if (queue && fence) wait_gpu();
        backbuffers.clear();
        swapchain.Reset();
        native_backbuffers.clear();
        native_swapchain.Reset();
        if (xefg) {
            xefg_swapchain_result_t r = xefgSwapChainDestroy(xefg);
            if (r != XEFG_SWAPCHAIN_RESULT_SUCCESS)
                fprintf(stderr, "[xefg] destroy: %s (%d)\n", fg_result_string(r), static_cast<int>(r));
        }
        if (xell) xellDestroyContext(xell);
        if (fence_event) CloseHandle(fence_event);
        if (hwnd) DestroyWindow(hwnd);
    }

    bool wait_gpu() {
        const uint64_t value = ++fence_value;
        if (FAILED(queue->Signal(fence.Get(), value))) return false;
        if (fence->GetCompletedValue() < value) {
            if (FAILED(fence->SetEventOnCompletion(value, fence_event))) return false;
            if (WaitForSingleObject(fence_event, INFINITE) != WAIT_OBJECT_0) return false;
        }
        return true;
    }

    bool reset_list() {
        return hr_ok(allocator->Reset(), "CommandAllocator::Reset") &&
               hr_ok(list->Reset(allocator.Get(), nullptr), "CommandList::Reset");
    }

    bool execute_list() {
        if (!hr_ok(list->Close(), "CommandList::Close")) return false;
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        return true;
    }
};

struct WindowCapture {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11Texture2D> staging;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice winrt_device{nullptr};
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{nullptr};
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool pool{nullptr};
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession session{nullptr};
    int width = 0;
    int height = 0;

    ~WindowCapture() {
        try {
            if (session) session.Close();
            if (pool) pool.Close();
        } catch (...) {
        }
    }

    bool initialize(IDXGIAdapter1* adapter, HWND hwnd, int capture_width, int capture_height) {
        width = capture_width;
        height = capture_height;
        if (!winrt::Windows::Graphics::Capture::GraphicsCaptureSession::IsSupported()) {
            fprintf(stderr, "[capture] Windows Graphics Capture is not supported\n");
            return false;
        }
        D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL selected{};
        HRESULT hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                                       D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels,
                                       static_cast<UINT>(std::size(levels)), D3D11_SDK_VERSION,
                                       &device, &selected, &context);
        if (!hr_ok(hr, "D3D11CreateDevice(capture)")) return false;

        ComPtr<IDXGIDevice> dxgi_device;
        if (!hr_ok(device.As(&dxgi_device), "Query IDXGIDevice(capture)")) return false;
        winrt::com_ptr<IInspectable> inspectable;
        if (!hr_ok(CreateDirect3D11DeviceFromDXGIDevice(dxgi_device.Get(), inspectable.put()),
                   "CreateDirect3D11DeviceFromDXGIDevice")) return false;
        winrt_device = inspectable.as<
            winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();

        auto factory = winrt::get_activation_factory<
            winrt::Windows::Graphics::Capture::GraphicsCaptureItem,
            IGraphicsCaptureItemInterop>();
        winrt::check_hresult(factory->CreateForWindow(
            hwnd,
            winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(),
            winrt::put_abi(item)));
        const auto item_size = item.Size();
        if (item_size.Width != width || item_size.Height != height) {
            fprintf(stderr, "[capture] unexpected window size %dx%d (wanted %dx%d)\n",
                    item_size.Width, item_size.Height, width, height);
            return false;
        }
        pool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
            winrt_device,
            winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            3, item_size);
        session = pool.CreateCaptureSession(item);
        session.IsCursorCaptureEnabled(false);
        session.StartCapture();
        Sleep(50);
        drain();
        fprintf(stderr, "[capture] window capture ready: %dx%d\n", width, height);
        return true;
    }

    void drain() {
        if (!pool) return;
        for (;;) {
            auto frame = pool.TryGetNextFrame();
            if (!frame) break;
            frame.Close();
        }
    }

    bool acquire_rgb(FILE* output, DWORD timeout_ms = 3000) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame frame{nullptr};
        while (std::chrono::steady_clock::now() < deadline) {
            frame = pool.TryGetNextFrame();
            if (frame) break;
            Sleep(1);
        }
        if (!frame) {
            fprintf(stderr, "[capture] timed out waiting for a presented frame\n");
            return false;
        }

        const auto content_size = frame.ContentSize();
        if (content_size.Width != width || content_size.Height != height) {
            fprintf(stderr, "[capture] unexpected content size %dx%d\n",
                    content_size.Width, content_size.Height);
            return false;
        }
        auto access = frame.Surface().as<
            ::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        ComPtr<ID3D11Texture2D> texture;
        if (!hr_ok(access->GetInterface(IID_PPV_ARGS(&texture)),
                   "capture surface GetInterface")) return false;
        D3D11_TEXTURE2D_DESC source_desc{};
        texture->GetDesc(&source_desc);
        if (source_desc.Width < static_cast<UINT>(width) ||
            source_desc.Height < static_cast<UINT>(height) ||
            source_desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
            fprintf(stderr, "[capture] unsupported surface: %ux%u format %u\n",
                    source_desc.Width, source_desc.Height,
                    static_cast<unsigned>(source_desc.Format));
            return false;
        }
        if (!staging) {
            D3D11_TEXTURE2D_DESC staging_desc = source_desc;
            staging_desc.Width = width;
            staging_desc.Height = height;
            staging_desc.BindFlags = 0;
            staging_desc.MiscFlags = 0;
            staging_desc.Usage = D3D11_USAGE_STAGING;
            staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            if (!hr_ok(device->CreateTexture2D(&staging_desc, nullptr, &staging),
                       "CreateTexture2D(capture staging)")) return false;
        }
        context->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, texture.Get(), 0, nullptr);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (!hr_ok(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped),
                   "Map(capture staging)")) return false;
        std::vector<uint8_t> row(static_cast<size_t>(width) * 3);
        bool ok = true;
        for (int y = 0; y < height && ok; ++y) {
            const auto* bgra = static_cast<const uint8_t*>(mapped.pData) +
                               static_cast<size_t>(y) * mapped.RowPitch;
            for (int x = 0; x < width; ++x) {
                row[x * 3 + 0] = bgra[x * 4 + 2];
                row[x * 3 + 1] = bgra[x * 4 + 1];
                row[x * 3 + 2] = bgra[x * 4 + 0];
            }
            if (output) ok = fwrite(row.data(), 1, row.size(), output) == row.size();
        }
        context->Unmap(staging.Get(), 0);
        frame.Close();
        return ok;
    }
};

void fg_log(const char* message, xefg_swapchain_logging_level_t level, void*) {
    const char* label = level == XEFG_SWAPCHAIN_LOGGING_LEVEL_ERROR ? "error" :
                        level == XEFG_SWAPCHAIN_LOGGING_LEVEL_WARNING ? "warning" :
                        level == XEFG_SWAPCHAIN_LOGGING_LEVEL_INFO ? "info" : "debug";
    fprintf(stderr, "[xefg:%s] %s\n", label, message);
}

bool init_runtime(const Args& args, Runtime& runtime) {
    if (!hr_ok(CreateDXGIFactory1(IID_PPV_ARGS(&runtime.factory)), "CreateDXGIFactory1")) return false;
    runtime.adapter = pick_adapter(runtime.factory.Get(), args.device);
    if (!runtime.adapter) {
        fprintf(stderr, "[d3d12] no suitable adapter\n");
        return false;
    }
    DXGI_ADAPTER_DESC1 adapter_desc{};
    runtime.adapter->GetDesc1(&adapter_desc);
    fwprintf(stderr, L"[d3d12] adapter: %ls\n", adapter_desc.Description);
    if (!hr_ok(D3D12CreateDevice(runtime.adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&runtime.device)), "D3D12CreateDevice")) return false;

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT queue_hr = runtime.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&runtime.queue));
    if (!hr_ok(queue_hr, "CreateCommandQueue")) {
        fprintf(stderr, "[d3d12] device removed reason: 0x%08lx\n",
                static_cast<unsigned long>(runtime.device->GetDeviceRemovedReason()));
        return false;
    }
    if (!hr_ok(runtime.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                       IID_PPV_ARGS(&runtime.allocator)),
               "CreateCommandAllocator")) return false;
    if (!hr_ok(runtime.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 runtime.allocator.Get(), nullptr,
                                                 IID_PPV_ARGS(&runtime.list)),
               "CreateCommandList")) return false;
    if (!hr_ok(runtime.list->Close(), "initial CommandList::Close")) return false;
    if (!hr_ok(runtime.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&runtime.fence)),
               "CreateFence")) return false;
    runtime.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!runtime.fence_event) return false;

    runtime.hwnd = create_present_window(args.width, args.height);
    if (!runtime.hwnd) return hr_ok(HRESULT_FROM_WIN32(GetLastError()), "CreateWindowExW");

    xefg_swapchain_version_t version{};
    if (!fg_ok(xefgSwapChainGetVersion(&version), "GetVersion")) return false;
    fprintf(stderr, "[xefg] libxess_fg.dll version %u.%u.%u\n", version.major, version.minor, version.patch);

    xell_result_t xr = xellD3D12CreateContext(runtime.device.Get(), &runtime.xell);
    if (xr != XELL_RESULT_SUCCESS || !runtime.xell) {
        fprintf(stderr, "[xell] create context failed: %d\n", static_cast<int>(xr));
        return false;
    }
    xell_sleep_params_t sleep{};
    sleep.bLowLatencyMode = 1;
    sleep.minimumIntervalUs = static_cast<uint32_t>(1000000.0 / (args.fps * 2.0) + 0.5);
    xr = xellSetSleepMode(runtime.xell, &sleep);
    if (xr != XELL_RESULT_SUCCESS) {
        fprintf(stderr, "[xell] set sleep mode failed: %d\n", static_cast<int>(xr));
        return false;
    }

    if (!fg_ok(xefgSwapChainD3D12CreateContext(runtime.device.Get(), &runtime.xefg),
               "CreateContext") || !runtime.xefg) return false;
    xefgSwapChainSetLoggingCallback(runtime.xefg,
                                    args.verbose ? XEFG_SWAPCHAIN_LOGGING_LEVEL_DEBUG
                                                 : XEFG_SWAPCHAIN_LOGGING_LEVEL_WARNING,
                                    fg_log, nullptr);
    if (!fg_ok(xefgSwapChainSetLatencyReduction(runtime.xefg, runtime.xell),
               "SetLatencyReduction")) return false;

    DXGI_SWAP_CHAIN_DESC1 swap_desc{};
    swap_desc.Width = static_cast<UINT>(args.width);
    swap_desc.Height = static_cast<UINT>(args.height);
    swap_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swap_desc.SampleDesc.Count = 1;
    swap_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swap_desc.BufferCount = kBufferCount;
    swap_desc.Scaling = DXGI_SCALING_STRETCH;
    swap_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swap_desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

    const uint32_t init_flags = (args.depth_dir || args.stream)
        ? XEFG_SWAPCHAIN_INIT_FLAG_INVERTED_DEPTH
        : XEFG_SWAPCHAIN_INIT_FLAG_NONE;
    xefg_swapchain_result_t build = xefgSwapChainD3D12BuildPipelines(
        runtime.xefg, nullptr, true, init_flags);
    if (build != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        fprintf(stderr, "[xefg] BuildPipelines: %s (%d), continuing\n",
                fg_result_string(build), static_cast<int>(build));

    xefg_swapchain_d3d12_init_params_t init{};
    init.initFlags = init_flags;
    init.maxInterpolatedFrames = 1;
    init.uiMode = args.ui_mask_dir ? XEFG_SWAPCHAIN_UI_MODE_BACKBUFFER_UITEXTURE
                                   : XEFG_SWAPCHAIN_UI_MODE_NONE;

    ComPtr<IDXGIFactory2> init_factory;
    RecordingFactory* recording_factory = nullptr;
    if (args.direct_capture) {
        recording_factory = new RecordingFactory(runtime.factory.Get());
        init_factory.Attach(recording_factory);
    } else if (!hr_ok(runtime.factory.As(&init_factory), "Query IDXGIFactory2")) {
        return false;
    }
    if (!fg_ok(xefgSwapChainD3D12InitFromSwapChainDesc(
                   runtime.xefg, runtime.hwnd, &swap_desc, nullptr, runtime.queue.Get(),
                   init_factory.Get(), &init), "InitFromSwapChainDesc")) return false;
    if (args.direct_capture) {
        IDXGISwapChain1* captured = recording_factory->captured_swapchain();
        if (!captured) {
            fprintf(stderr, "[capture] XeSS-FG did not create an HWND swap chain\n");
            return false;
        }
        if (!hr_ok(captured->QueryInterface(IID_PPV_ARGS(&runtime.native_swapchain)),
                   "Query native IDXGISwapChain4")) return false;
    }
    init_factory.Reset();
    if (!fg_ok(xefgSwapChainD3D12GetSwapChainPtr(runtime.xefg,
                                                 IID_PPV_ARGS(&runtime.swapchain)),
               "GetSwapChainPtr")) return false;
    if (!fg_ok(xefgSwapChainEnableDebugFeature(
                   runtime.xefg, XEFG_SWAPCHAIN_DEBUG_FEATURE_SHOW_ONLY_INTERPOLATION,
                   1, nullptr), "EnableDebugFeature(SHOW_ONLY_INTERPOLATION)")) return false;
    if (!fg_ok(xefgSwapChainSetEnabled(runtime.xefg, 1), "SetEnabled")) return false;

    DXGI_SWAP_CHAIN_DESC1 actual_desc{};
    runtime.swapchain->GetDesc1(&actual_desc);
    fprintf(stderr, "[xefg] proxy swap chain: %ux%u, %u buffers, format %u\n",
            actual_desc.Width, actual_desc.Height, actual_desc.BufferCount,
            static_cast<unsigned>(actual_desc.Format));
    for (UINT i = 0; i < actual_desc.BufferCount; ++i) {
        ComPtr<ID3D12Resource> buffer;
        if (!hr_ok(runtime.swapchain->GetBuffer(i, IID_PPV_ARGS(&buffer)), "SwapChain::GetBuffer"))
            return false;
        runtime.backbuffers.push_back(buffer);
    }
    if (args.direct_capture) {
        DXGI_SWAP_CHAIN_DESC1 native_desc{};
        if (!hr_ok(runtime.native_swapchain->GetDesc1(&native_desc),
                   "NativeSwapChain::GetDesc1")) return false;
        fprintf(stderr, "[capture] direct native swap chain: %ux%u, %u buffers\n",
                native_desc.Width, native_desc.Height, native_desc.BufferCount);
        for (UINT i = 0; i < native_desc.BufferCount; ++i) {
            ComPtr<ID3D12Resource> buffer;
            if (!hr_ok(runtime.native_swapchain->GetBuffer(i, IID_PPV_ARGS(&buffer)),
                       "NativeSwapChain::GetBuffer")) return false;
            runtime.native_backbuffers.push_back(buffer);
        }
    }

    runtime.velocity = create_texture(runtime.device.Get(), args.width, args.height,
                                      DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST,
                                      L"XeSS-FG velocity");
    runtime.depth = create_texture(runtime.device.Get(), args.width, args.height,
                                   DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST,
                                   L"XeSS-FG depth");
    if (args.ui_mask_dir)
        runtime.ui = create_texture(runtime.device.Get(), args.width, args.height,
                                    DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_STATE_COPY_DEST,
                                    L"XeSS-FG UI texture");
    if (!runtime.velocity || !runtime.depth || (args.ui_mask_dir && !runtime.ui)) return false;

    runtime.color_pitch = align_up(static_cast<uint32_t>(args.width) * 4,
                                   D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    runtime.velocity_pitch = align_up(static_cast<uint32_t>(args.width) * 4,
                                      D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    runtime.depth_pitch = align_up(static_cast<uint32_t>(args.width) * 4,
                                   D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    runtime.ui_pitch = runtime.color_pitch;
    runtime.readback_pitch = runtime.color_pitch;
    const uint64_t color_bytes = static_cast<uint64_t>(runtime.color_pitch) * args.height;
    runtime.velocity_offset = align_up(static_cast<uint32_t>(color_bytes),
                                       D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    const uint64_t velocity_bytes = static_cast<uint64_t>(runtime.velocity_pitch) * args.height;
    runtime.depth_offset = align_up(static_cast<uint32_t>(runtime.velocity_offset + velocity_bytes),
                                    D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    const uint64_t depth_bytes = static_cast<uint64_t>(runtime.depth_pitch) * args.height;
    runtime.ui_offset = align_up(static_cast<uint32_t>(runtime.depth_offset + depth_bytes),
                                 D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    const uint64_t upload_bytes = args.ui_mask_dir
        ? runtime.ui_offset + static_cast<uint64_t>(runtime.ui_pitch) * args.height
        : runtime.depth_offset + depth_bytes;
    runtime.upload = create_buffer(runtime.device.Get(), upload_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                   D3D12_RESOURCE_STATE_GENERIC_READ, L"XeSS-FG upload");
    runtime.readback = create_buffer(runtime.device.Get(), color_bytes, D3D12_HEAP_TYPE_READBACK,
                                     D3D12_RESOURCE_STATE_COPY_DEST, L"XeSS-FG readback");
    return runtime.upload && runtime.readback;
}

bool upload_frame(Runtime& runtime, const Args& args, const uint8_t* rgb, const float* motion,
                  const float* depth_input, const uint8_t* ui_mask,
                  ID3D12Resource* backbuffer, bool first_frame) {
    void* mapped = nullptr;
    if (!hr_ok(runtime.upload->Map(0, nullptr, &mapped), "upload Map")) return false;
    auto* bytes = static_cast<uint8_t*>(mapped);
    for (int y = 0; y < args.height; ++y) {
        uint8_t* dst = bytes + static_cast<size_t>(y) * runtime.color_pitch;
        const uint8_t* src = rgb + static_cast<size_t>(y) * args.width * 3;
        for (int x = 0; x < args.width; ++x) {
            dst[x * 4 + 0] = src[x * 3 + 0];
            dst[x * 4 + 1] = src[x * 3 + 1];
            dst[x * 4 + 2] = src[x * 3 + 2];
            dst[x * 4 + 3] = 255;
        }
    }
    if (args.ui_mask_dir) {
        if (!ui_mask) { runtime.upload->Unmap(0, nullptr); return false; }
        for (int y = 0; y < args.height; ++y) {
            uint8_t* dst = bytes + runtime.ui_offset + static_cast<size_t>(y) * runtime.ui_pitch;
            const uint8_t* color = rgb + static_cast<size_t>(y) * args.width * 3;
            const uint8_t* alpha = ui_mask + static_cast<size_t>(y) * args.width;
            for (int x = 0; x < args.width; ++x) {
                const uint32_t a = alpha[x];
                dst[x * 4 + 0] = static_cast<uint8_t>((color[x * 3 + 0] * a + 127) / 255);
                dst[x * 4 + 1] = static_cast<uint8_t>((color[x * 3 + 1] * a + 127) / 255);
                dst[x * 4 + 2] = static_cast<uint8_t>((color[x * 3 + 2] * a + 127) / 255);
                dst[x * 4 + 3] = static_cast<uint8_t>(a);
            }
        }
    }
    for (int y = 0; y < args.height; ++y) {
        auto* dst = reinterpret_cast<uint16_t*>(bytes + runtime.velocity_offset +
                                               static_cast<size_t>(y) * runtime.velocity_pitch);
        const float* src = motion + static_cast<size_t>(y) * args.width * 2;
        for (int x = 0; x < args.width * 2; ++x) dst[x] = f32_to_f16(src[x]);
    }
    for (int y = 0; y < args.height; ++y) {
        auto* dst = reinterpret_cast<float*>(bytes + runtime.depth_offset +
                                             static_cast<size_t>(y) * runtime.depth_pitch);
        if (depth_input) {
            const float* src = depth_input + static_cast<size_t>(y) * args.width;
            std::copy(src, src + args.width, dst);
        } else {
            std::fill(dst, dst + args.width, 0.5f);
        }
    }
    runtime.upload->Unmap(0, nullptr);

    if (!runtime.reset_list()) return false;
    transition(runtime.list.Get(), backbuffer, D3D12_RESOURCE_STATE_PRESENT,
               D3D12_RESOURCE_STATE_COPY_DEST);
    if (!first_frame) {
        transition(runtime.list.Get(), runtime.velocity.Get(),
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        transition(runtime.list.Get(), runtime.depth.Get(),
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        if (args.ui_mask_dir)
            transition(runtime.list.Get(), runtime.ui.Get(),
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_COPY_DEST);
    }

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = runtime.upload.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint.Width = args.width;
    src.PlacedFootprint.Footprint.Height = args.height;
    src.PlacedFootprint.Footprint.Depth = 1;

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

    src.PlacedFootprint.Offset = 0;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    src.PlacedFootprint.Footprint.RowPitch = runtime.color_pitch;
    dst.pResource = backbuffer;
    runtime.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    if (args.ui_mask_dir) {
        src.PlacedFootprint.Offset = runtime.ui_offset;
        src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        src.PlacedFootprint.Footprint.RowPitch = runtime.ui_pitch;
        dst.pResource = runtime.ui.Get();
        runtime.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }

    src.PlacedFootprint.Offset = runtime.velocity_offset;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16_FLOAT;
    src.PlacedFootprint.Footprint.RowPitch = runtime.velocity_pitch;
    dst.pResource = runtime.velocity.Get();
    runtime.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    src.PlacedFootprint.Offset = runtime.depth_offset;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
    src.PlacedFootprint.Footprint.RowPitch = runtime.depth_pitch;
    dst.pResource = runtime.depth.Get();
    runtime.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    transition(runtime.list.Get(), backbuffer, D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_PRESENT);
    transition(runtime.list.Get(), runtime.velocity.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(runtime.list.Get(), runtime.depth.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (args.ui_mask_dir)
        transition(runtime.list.Get(), runtime.ui.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return runtime.execute_list();
}

bool readback_rgb(Runtime& runtime, const Args& args, ID3D12Resource* texture, FILE* output) {
    if (!runtime.reset_list()) return false;
    transition(runtime.list.Get(), texture, D3D12_RESOURCE_STATE_PRESENT,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = runtime.readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dst.PlacedFootprint.Footprint.Width = args.width;
    dst.PlacedFootprint.Footprint.Height = args.height;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = runtime.readback_pitch;
    runtime.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    transition(runtime.list.Get(), texture, D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_PRESENT);
    if (!runtime.execute_list() || !runtime.wait_gpu()) return false;

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(runtime.readback_pitch) * args.height};
    if (!hr_ok(runtime.readback->Map(0, &read_range, &mapped), "readback Map")) return false;
    std::vector<uint8_t> row(static_cast<size_t>(args.width) * 3);
    const auto* rgba = static_cast<const uint8_t*>(mapped);
    bool ok = true;
    for (int y = 0; y < args.height && ok; ++y) {
        const uint8_t* src_row = rgba + static_cast<size_t>(y) * runtime.readback_pitch;
        for (int x = 0; x < args.width; ++x) {
            row[x * 3 + 0] = src_row[x * 4 + 0];
            row[x * 3 + 1] = src_row[x * 4 + 1];
            row[x * 3 + 2] = src_row[x * 4 + 2];
        }
        ok = fwrite(row.data(), 1, row.size(), output) == row.size();
    }
    D3D12_RANGE written{0, 0};
    runtime.readback->Unmap(0, &written);
    return ok;
}

bool dump_proxy_buffers(Runtime& runtime, const Args& args, int frame_index) {
    if (!args.dump_buffers || frame_index > 2) return true;
    CreateDirectoryA(args.dump_buffers, nullptr);
    for (size_t i = 0; i < runtime.backbuffers.size(); ++i) {
        char path[1024];
        snprintf(path, sizeof(path), "%s\\present_%03d_buffer_%zu.rgb", args.dump_buffers,
                 frame_index, i);
        FILE* file = fopen(path, "wb");
        if (!file) return false;
        const bool ok = readback_rgb(runtime, args, runtime.backbuffers[i].Get(), file);
        fclose(file);
        if (!ok) return false;
    }
    return true;
}

struct StreamFrame {
    std::vector<uint8_t> color;
    std::vector<float> motion;
    std::vector<float> depth;
    std::vector<uint8_t> ui_mask;
    uint32_t flags = 0;
};

// Returns 1 for a frame, 0 for EOS, -1 for malformed/truncated input.
int read_stream_frame(FILE* input, XessSharedRingReader* ring, const Args& args,
                      int expected_index, StreamFrame& frame) {
    std::vector<uint8_t> packet;
    StreamHeader header{};
    if (ring) {
        if (!ring->read(packet) || packet.size() < sizeof(header)) {
            fprintf(stderr, "[shm] missing packet at frame %d\n", expected_index);
            return -1;
        }
        memcpy(&header, packet.data(), sizeof(header));
    } else if (!read_exact(input, &header, sizeof(header))) {
        fprintf(stderr, "[stream] truncated header at frame %d\n", expected_index);
        return -1;
    }
    if (memcmp(header.magic, "XSPK", 4) || header.version != 1 ||
        header.header_size != sizeof(StreamHeader)) {
        fprintf(stderr, "[stream] invalid protocol header\n");
        return -1;
    }
    if (header.flags & kStreamFlagEos) {
        if (header.frame_index != static_cast<uint32_t>(expected_index) ||
            header.color_bytes || header.motion_bytes || header.depth_bytes ||
            header.mask_bytes) return -1;
        return 0;
    }
    const size_t pixels = static_cast<size_t>(args.width) * args.height;
    const uint64_t total = static_cast<uint64_t>(header.color_bytes) +
                           header.motion_bytes + header.depth_bytes + header.mask_bytes;
    if (total > 512ull * 1024 * 1024 ||
        header.frame_index != static_cast<uint32_t>(expected_index) ||
        header.width != static_cast<uint32_t>(args.width) ||
        header.height != static_cast<uint32_t>(args.height) || header.pixel_format != 1 ||
        header.color_bytes != pixels * 3 ||
        header.motion_bytes != pixels * 2 * sizeof(float) ||
        header.depth_bytes != pixels * sizeof(float) ||
        (args.ui_mask_dir ? header.mask_bytes != pixels : header.mask_bytes != 0)) {
        fprintf(stderr, "[stream] metadata/payload mismatch at frame %d\n", expected_index);
        return -1;
    }
    std::vector<uint8_t> payload(static_cast<size_t>(total));
    if (ring) {
        if (packet.size() != sizeof(header) + payload.size()) {
            fprintf(stderr, "[shm] packet length mismatch at frame %d\n", expected_index);
            return -1;
        }
        memcpy(payload.data(), packet.data() + sizeof(header), payload.size());
    } else if (!read_exact(input, payload.data(), payload.size())) {
        fprintf(stderr, "[stream] truncated/corrupt payload at frame %d\n", expected_index);
        return -1;
    }
    if (crc32_bytes(payload.data(), payload.size()) != header.checksum) {
        fprintf(stderr, "[stream] checksum mismatch at frame %d\n", expected_index);
        return -1;
    }
    size_t offset = 0;
    frame.color.assign(payload.begin(), payload.begin() + header.color_bytes);
    offset += header.color_bytes;
    frame.motion.resize(pixels * 2);
    memcpy(frame.motion.data(), payload.data() + offset, header.motion_bytes);
    offset += header.motion_bytes;
    frame.depth.resize(pixels);
    memcpy(frame.depth.data(), payload.data() + offset, header.depth_bytes);
    offset += header.depth_bytes;
    frame.ui_mask.clear();
    if (header.mask_bytes)
        frame.ui_mask.assign(payload.begin() + offset, payload.begin() + offset + header.mask_bytes);
    frame.flags = header.flags;
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    Args args;
    if (!parse_args(argc, argv, args)) {
        fprintf(stderr,
                "Usage: xess-fg.exe [--stream [--shm-name NAME --shm-slots N --shm-slot-size N] "
                "| --frames in.raw --mv mvs --out generated.raw] "
                "--width W --height H --frames-count N --fps FPS "
                "[--depth depths] [--device N] [--verbose] [--dump-buffers DIR] "
                "[--capture-mode direct|window] [--allow-overlay]\n");
        return 2;
    }
    if (args.stream) {
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
    }
    if (!args.direct_capture && !args.allow_overlay && process_running(L"RTSS.exe")) {
        fprintf(stderr,
                "[capture] RTSS.exe is running. Its on-screen display is drawn into captured "
                "XeSS-FG frames. Exit RivaTuner Statistics Server or create an OSD-disabled "
                "profile for xess-fg.exe, then retry. Use --allow-overlay only if that profile "
                "is already active.\n");
        return 1;
    }

    Runtime runtime;
    if (!init_runtime(args, runtime)) return 1;
    WindowCapture capture;
    if (args.direct_capture) {
        fprintf(stderr, "[capture] mode=direct (native swap-chain readback)\n");
    } else {
        try {
            if (!capture.initialize(runtime.adapter.Get(), runtime.hwnd,
                                    args.width, args.height)) return 1;
        } catch (const winrt::hresult_error& error) {
            fprintf(stderr, "[capture] initialization failed: HRESULT 0x%08lx\n",
                    static_cast<unsigned long>(error.code().value));
            return 1;
        }
    }

    FILE* input = args.stream ? stdin : fopen(args.frames, "rb");
    if (!input) {
        fprintf(stderr, "[io] cannot open %s\n", args.frames);
        return 1;
    }
    FILE* output = args.stream ? stdout : fopen(args.output, "wb");
    if (!output) {
        fprintf(stderr, "[io] cannot open %s\n", args.output);
        if (!args.stream) fclose(input);
        return 1;
    }
    XessSharedRingReader shared_ring;
    XessSharedRingReader* shared_ring_ptr = nullptr;
    if (args.shm_name) {
        if (!shared_ring.open(args.shm_name, args.shm_slots, args.shm_slot_size)) {
            if (!args.stream) { fclose(output); fclose(input); }
            return 1;
        }
        shared_ring_ptr = &shared_ring;
    }

    const size_t pixel_count = static_cast<size_t>(args.width) * args.height;
    std::vector<uint8_t> rgb(pixel_count * 3);
    std::vector<float> motion(pixel_count * 2);
    std::vector<float> depth_values(pixel_count);
    std::vector<uint8_t> ui_mask(pixel_count);
    int generated_count = 0;
    bool failed = false;
    StreamFrame stream_frame;
    std::vector<uint8_t> reset_history(static_cast<size_t>(args.frame_count), 0);
    reset_history[0] = 1;
    if (!args.stream && args.reset_frames) {
        FILE* reset_file = fopen(args.reset_frames, "r");
        if (!reset_file) {
            fprintf(stderr, "[io] cannot open reset-frame file %s\n", args.reset_frames);
            fclose(output); fclose(input);
            return 1;
        }
        int index = 0;
        while (fscanf(reset_file, "%d", &index) == 1)
            if (index >= 0 && index < args.frame_count) reset_history[static_cast<size_t>(index)] = 1;
        fclose(reset_file);
    }

    for (int frame = 0; frame < args.frame_count; ++frame) {
        MSG message{};
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        bool reset = reset_history[static_cast<size_t>(frame)] != 0;
        const float* depth_input = nullptr;
        if (args.stream) {
            const int stream_result = read_stream_frame(input, shared_ring_ptr, args,
                                                        frame, stream_frame);
            if (stream_result != 1) {
                fprintf(stderr, "[stream] expected frame %d\n", frame);
                failed = true;
                break;
            }
            rgb = stream_frame.color;
            motion = stream_frame.motion;
            depth_values = stream_frame.depth;
            depth_input = depth_values.data();
            if (args.ui_mask_dir) ui_mask = stream_frame.ui_mask;
            reset = (stream_frame.flags & (kStreamFlagReset | kStreamFlagSceneCut)) != 0;
        } else {
            if (!read_exact(input, rgb.data(), rgb.size())) {
                fprintf(stderr, "[io] input frame %d is incomplete\n", frame);
                failed = true;
                break;
            }
            char mv_path[1024];
            snprintf(mv_path, sizeof(mv_path), "%s\\mv_%06d.bin", args.mv_dir, frame);
            FILE* mv_file = fopen(mv_path, "rb");
            if (!mv_file || !read_exact(mv_file, motion.data(), motion.size() * sizeof(float))) {
                fprintf(stderr, "[io] motion-vector file is missing or incomplete: %s\n", mv_path);
                if (mv_file) fclose(mv_file);
                failed = true;
                break;
            }
            fclose(mv_file);
        }

        if (!args.stream && args.depth_dir) {
            char depth_path[1024];
            snprintf(depth_path, sizeof(depth_path), "%s\\depth_%06d.bin", args.depth_dir, frame);
            FILE* depth_file = fopen(depth_path, "rb");
            if (!depth_file || !read_exact(depth_file, depth_values.data(),
                                           depth_values.size() * sizeof(float))) {
                fprintf(stderr, "[io] depth file is missing or incomplete: %s\n", depth_path);
                if (depth_file) fclose(depth_file);
                failed = true;
                break;
            }
            fclose(depth_file);
            depth_input = depth_values.data();
        }
        if (!args.stream && args.ui_mask_dir) {
            char mask_path[1024];
            snprintf(mask_path, sizeof(mask_path), "%s\\mask_%06d.bin", args.ui_mask_dir, frame);
            FILE* mask_file = fopen(mask_path, "rb");
            if (!mask_file || !read_exact(mask_file, ui_mask.data(), ui_mask.size())) {
                fprintf(stderr, "[io] UI mask file is missing or incomplete: %s\n", mask_path);
                if (mask_file) fclose(mask_file);
                failed = true;
                break;
            }
            fclose(mask_file);
        }

        const uint32_t present_id = static_cast<uint32_t>(frame + 1);
        if (xellSleep(runtime.xell, present_id) != XELL_RESULT_SUCCESS ||
            xellAddMarkerData(runtime.xell, present_id, XELL_SIMULATION_START) != XELL_RESULT_SUCCESS ||
            xellAddMarkerData(runtime.xell, present_id, XELL_SIMULATION_END) != XELL_RESULT_SUCCESS ||
            xellAddMarkerData(runtime.xell, present_id, XELL_RENDERSUBMIT_START) != XELL_RESULT_SUCCESS) {
            fprintf(stderr, "[xell] marker/sleep failure on frame %d\n", frame);
            failed = true;
            break;
        }

        const UINT buffer_index = runtime.swapchain->GetCurrentBackBufferIndex();
        if (buffer_index >= runtime.backbuffers.size() ||
            !upload_frame(runtime, args, rgb.data(), motion.data(), depth_input,
                          args.ui_mask_dir ? ui_mask.data() : nullptr,
                          runtime.backbuffers[buffer_index].Get(), frame == 0)) {
            failed = true;
            break;
        }

        xefg_swapchain_d3d12_resource_data_t velocity{};
        velocity.type = XEFG_SWAPCHAIN_RES_MOTION_VECTOR;
        velocity.validity = XEFG_SWAPCHAIN_RV_UNTIL_NEXT_PRESENT;
        velocity.resourceSize = {static_cast<uint32_t>(args.width), static_cast<uint32_t>(args.height)};
        velocity.pResource = runtime.velocity.Get();
        velocity.incomingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if (!fg_ok(xefgSwapChainD3D12TagFrameResource(runtime.xefg, runtime.list.Get(),
                                                       present_id, &velocity),
                   "TagFrameResource(MOTION_VECTOR)")) {
            failed = true;
            break;
        }

        xefg_swapchain_d3d12_resource_data_t depth{};
        depth.type = XEFG_SWAPCHAIN_RES_DEPTH;
        depth.validity = XEFG_SWAPCHAIN_RV_UNTIL_NEXT_PRESENT;
        depth.resourceSize = {static_cast<uint32_t>(args.width), static_cast<uint32_t>(args.height)};
        depth.pResource = runtime.depth.Get();
        depth.incomingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if (!fg_ok(xefgSwapChainD3D12TagFrameResource(runtime.xefg, runtime.list.Get(),
                                                       present_id, &depth),
                   "TagFrameResource(DEPTH)")) {
            failed = true;
            break;
        }

        if (args.ui_mask_dir) {
            xefg_swapchain_d3d12_resource_data_t ui{};
            ui.type = XEFG_SWAPCHAIN_RES_UI;
            ui.validity = XEFG_SWAPCHAIN_RV_UNTIL_NEXT_PRESENT;
            ui.resourceSize = {static_cast<uint32_t>(args.width), static_cast<uint32_t>(args.height)};
            ui.pResource = runtime.ui.Get();
            ui.incomingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            if (!fg_ok(xefgSwapChainD3D12TagFrameResource(runtime.xefg, runtime.list.Get(),
                                                           present_id, &ui),
                       "TagFrameResource(UI)")) {
                failed = true;
                break;
            }
        }

        xefg_swapchain_frame_constant_data_t constants{};
        for (int i = 0; i < 16; ++i) {
            constants.viewMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
            constants.projectionMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        }
        constants.motionVectorScaleX = 1.0f;
        constants.motionVectorScaleY = 1.0f;
        constants.resetHistory = reset ? 1u : 0u;
        constants.frameRenderTime = static_cast<float>(1000.0 / args.fps);
        if (!fg_ok(xefgSwapChainTagFrameConstants(runtime.xefg, present_id, &constants),
                   "TagFrameConstants") ||
            !fg_ok(xefgSwapChainSetPresentId(runtime.xefg, present_id), "SetPresentId")) {
            failed = true;
            break;
        }

        if (xellAddMarkerData(runtime.xell, present_id, XELL_RENDERSUBMIT_END) != XELL_RESULT_SUCCESS ||
            xellAddMarkerData(runtime.xell, present_id, XELL_PRESENT_START) != XELL_RESULT_SUCCESS) {
            fprintf(stderr, "[xell] pre-present marker failure on frame %d\n", frame);
            failed = true;
            break;
        }
        UINT native_index_before = 0;
        UINT native_present_count_before = 0;
        if (args.direct_capture) {
            native_index_before = runtime.native_swapchain->GetCurrentBackBufferIndex();
            runtime.native_swapchain->GetLastPresentCount(&native_present_count_before);
        } else {
            capture.drain();
        }
        DXGI_SWAP_CHAIN_DESC1 desc{};
        runtime.swapchain->GetDesc1(&desc);
        const bool allow_tearing = (desc.Flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) != 0;
        const UINT present_flags = allow_tearing ? DXGI_PRESENT_ALLOW_TEARING : 0;
        HRESULT present_hr = runtime.swapchain->Present(allow_tearing ? 0 : 1, present_flags);
        xellAddMarkerData(runtime.xell, present_id, XELL_PRESENT_END);
        if (FAILED(present_hr)) {
            hr_ok(present_hr, "SwapChain::Present");
            failed = true;
            break;
        }

        xefg_swapchain_present_status_t status{};
        xefg_swapchain_result_t status_result =
            xefgSwapChainGetLastPresentStatus(runtime.xefg, &status);
        if (!fg_ok(status_result, "GetLastPresentStatus")) {
            failed = true;
            break;
        }
        if (!runtime.wait_gpu()) {
            failed = true;
            break;
        }
        UINT native_index_after = 0;
        UINT native_present_count_after = 0;
        if (args.direct_capture) {
            native_index_after = runtime.native_swapchain->GetCurrentBackBufferIndex();
            runtime.native_swapchain->GetLastPresentCount(&native_present_count_after);
            if (args.verbose) {
                fprintf(stderr,
                        "[capture] frame %d: native index %u->%u, present count %u->%u\n",
                        frame, native_index_before, native_index_after,
                        native_present_count_before, native_present_count_after);
            }
        }
        fprintf(stderr, "[xefg] frame %d: framesPresented=%u, enabled=%u, generation=%s (%d), reset=%u\n",
                frame, status.framesPresented, status.isFrameGenEnabled,
                fg_result_string(status.frameGenResult), static_cast<int>(status.frameGenResult),
                reset ? 1u : 0u);

        const bool output_needed = frame > 0;
        if (output_needed && status.frameGenResult < 0) {
            fprintf(stderr, "[xefg] frame generation failed on frame %d: %s (%d)\n",
                    frame, fg_result_string(status.frameGenResult),
                    static_cast<int>(status.frameGenResult));
            failed = true;
            break;
        }
        const bool force_fallback = output_needed && reset;
        if (status.framesPresented > 0) {
            bool capture_ok = true;
            if (args.direct_capture) {
                const UINT native_count = static_cast<UINT>(runtime.native_backbuffers.size());
                if (!native_count) {
                    fprintf(stderr, "[capture] native swap chain has no buffers\n");
                    capture_ok = false;
                } else if (output_needed && !force_fallback) {
                    const UINT last_presented =
                        (native_index_after + native_count - 1) % native_count;
                    capture_ok = readback_rgb(
                        runtime, args, runtime.native_backbuffers[last_presented].Get(), output);
                }
            } else {
                capture_ok = capture.acquire_rgb(
                    output_needed && !force_fallback ? output : nullptr);
            }
            if (!capture_ok) {
                failed = true;
                break;
            }
            if (force_fallback && fwrite(rgb.data(), 1, rgb.size(), output) != rgb.size()) {
                fprintf(stderr, "[io] failed to write reset fallback frame %d\n", frame);
                failed = true;
                break;
            }
        } else if (output_needed) {
            // Scene-change/history warnings can legitimately suppress interpolation.  Preserve
            // CFR output with the current application frame, matching the proxy's fallback rule.
            if (fwrite(rgb.data(), 1, rgb.size(), output) != rgb.size()) {
                fprintf(stderr, "[io] failed to write fallback frame %d\n", frame);
                failed = true;
                break;
            }
            fprintf(stderr, "[xefg] frame %d used current-frame fallback (%s)\n",
                    frame, fg_result_string(status.frameGenResult));
        }

        // Stream mode directly emits f0,G1,f1,G2,f2... so no full-size
        // interleaved raw file or Python-side source-frame cache is needed.
        if (args.stream && fwrite(rgb.data(), 1, rgb.size(), output) != rgb.size()) {
            fprintf(stderr, "[io] failed to write source frame %d\n", frame);
            failed = true;
            break;
        }

        if (!dump_proxy_buffers(runtime, args, frame)) {
            failed = true;
            break;
        }
        if (output_needed) {
            ++generated_count;
        }
        if (args.verbose || (frame + 1) % 25 == 0)
            fprintf(stderr, "[xefg] processed %d/%d input frames\n", frame + 1, args.frame_count);
    }

    if (!failed && args.stream) {
        const int eos_result = read_stream_frame(input, shared_ring_ptr, args,
                                                 args.frame_count, stream_frame);
        if (eos_result != 0) {
            fprintf(stderr, "[stream] missing EOS or extra frame\n");
            failed = true;
        }
        fflush(output);
    }
    if (!args.stream) { fclose(output); fclose(input); }
    fprintf(stderr, "[xefg] wrote %d generated-frame candidates\n", generated_count);
    return failed ? 1 : 0;
}
