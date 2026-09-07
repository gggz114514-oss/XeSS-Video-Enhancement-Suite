#pragma once
#include "gpu_frame_contract.h"
#include <d3d11.h>

namespace xess_gpu {
using MotionProviderFactory=MotionProvider*(*)(const char* backend);
// Optional compile-time wrapper registration. A factory returns a new owned
// provider; the native main initializes it once and destroys it after drain.
inline MotionProviderFactory native_motion_factory=nullptr;
// Both handles reference the same NV12 allocation. D3D11 is required by the
// selected OpenVINO remote-tensor API; texture ownership alone does not pin a
// decoder/capture pool entry. producer_pool_owner must hold the actual lease.
struct ExternalGpuFrame {
    FrameLease frame;
    ComPtr<ID3D11Texture2D> nv12_d3d11;
};

inline bool accept_external_gpu_frame(const ExternalGpuFrame& input,
                                      ID3D12Device* device,
                                      ID3D12CommandQueue* consumer_queue,
                                      Extent source, std::string& error) {
    if (!device || !consumer_queue || !input.frame.color.resource ||
        !input.nv12_d3d11 || !input.frame.producer_pool_owner) {
        error="external_input_missing_texture_or_pool_lease"; return false;
    }
    if (!same_adapter(input.frame.metadata.adapter_luid,device->GetAdapterLuid())) {
        error="external_input_adapter_mismatch";return false;
    }
    ComPtr<ID3D11Device> native_device;input.nv12_d3d11->GetDevice(&native_device);
    ComPtr<IDXGIDevice> dxgi;ComPtr<IDXGIAdapter> adapter;DXGI_ADAPTER_DESC adapter_desc{};
    ComPtr<ID3D12Device> resource_device;
    if(FAILED(native_device.As(&dxgi)) || FAILED(dxgi->GetAdapter(&adapter)) || FAILED(adapter->GetDesc(&adapter_desc)) ||
       FAILED(input.frame.color.resource->GetDevice(IID_PPV_ARGS(&resource_device))) ||
       !same_adapter(adapter_desc.AdapterLuid,device->GetAdapterLuid()) ||
       !same_adapter(resource_device->GetAdapterLuid(),device->GetAdapterLuid())) {
        error="external_input_actual_device_mismatch";return false;
    }
    const auto d=input.frame.color.resource->GetDesc();
    const auto& crop=input.frame.color.valid;
    if (d.Format!=DXGI_FORMAT_NV12 || d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        input.frame.color.format!=DXGI_FORMAT_NV12 || crop.width!=source.width || crop.height!=source.height ||
        crop.x!=0 || crop.y!=0 || crop.width>d.Width || crop.height>d.Height) {
        error="external_input_unsupported_format_or_crop";return false;
    }
    if(input.frame.color.state!=D3D12_RESOURCE_STATE_COMMON) {
        error="external_input_requires_common_handoff_state";return false;
    }
    if(input.frame.metadata.source_frame_id==kNoFrame ||
       (input.frame.metadata.pts_known && input.frame.metadata.original_time_base.denominator<=0)) {
        error="external_input_invalid_identity_or_timebase";return false;
    }
    if(input.frame.produced.value) {
        if(!input.frame.produced.fence || FAILED(consumer_queue->Wait(input.frame.produced.fence.Get(),input.frame.produced.value))) {
            error="external_input_producer_fence_wait";return false;
        }
    }
    return true;
}
} // namespace xess_gpu
