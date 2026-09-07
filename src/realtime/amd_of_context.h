#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <ffx_dx12.h>
#include <ffx_opticalflow.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Official FidelityFX Optical Flow context. All work is recorded onto the
// caller's ordered list; this wrapper never submits a queue or waits the CPU.
class AmdOfContext {
    std::unique_ptr<FfxOpticalflowContext> context_;
    std::vector<uint8_t> scratch_;
public:
    Microsoft::WRL::ComPtr<ID3D12Resource> flow,scene;
    uint64_t calls=0,resets=0;
    static void check(FfxErrorCode code,const char* stage) {
        if(code!=FFX_OK)throw std::runtime_error(std::string(stage)+" FFX="+std::to_string(code));
    }
    static Microsoft::WRL::ComPtr<ID3D12Resource> texture(ID3D12Device* device,UINT width,UINT height,DXGI_FORMAT format) {
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width=width;desc.Height=height;desc.DepthOrArraySize=1;desc.MipLevels=1;
        desc.Format=format;desc.SampleDesc.Count=1;desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        if(FAILED(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&resource))))
            throw std::runtime_error("amd_of_texture_allocation");
        return resource;
    }
    AmdOfContext(ID3D12Device* device,UINT width,UINT height) {
        scratch_.resize(ffxGetScratchMemorySizeDX12(1));
        FfxOpticalflowContextDescription desc{};desc.resolution={width,height};
        check(ffxGetInterfaceDX12(&desc.backendInterface,ffxGetDeviceDX12(device),
            scratch_.data(),scratch_.size(),1),"amd_interface");
        context_=std::make_unique<FfxOpticalflowContext>();
        check(ffxOpticalflowContextCreate(context_.get(),&desc),"amd_create");
        flow=texture(device,(width+7)/8,(height+7)/8,DXGI_FORMAT_R16G16_SINT);
        scene=texture(device,3,1,DXGI_FORMAT_R32_UINT);
    }
    ~AmdOfContext(){if(context_)ffxOpticalflowContextDestroy(context_.get());}
    void record(ID3D12GraphicsCommandList* list,ID3D12Resource* rgba,bool reset) {
        FfxOpticalflowDispatchDescription desc{};desc.commandList=ffxGetCommandListDX12(list);
        desc.color=ffxGetResourceDX12(rgba,ffxGetResourceDescriptionDX12(rgba),L"shared core source RGBA",FFX_API_RESOURCE_STATE_COMPUTE_READ);
        desc.opticalFlowVector=ffxGetResourceDX12(flow.Get(),ffxGetResourceDescriptionDX12(flow.Get()),L"official signed source-pixel grid",FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        desc.opticalFlowSCD=ffxGetResourceDX12(scene.Get(),ffxGetResourceDescriptionDX12(scene.Get()),L"official scene data",FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        desc.reset=reset;desc.backbufferTransferFunction=0;desc.minMaxLuminance={0.f,1.f};
        check(ffxOpticalflowContextDispatch(context_.get(),&desc),"amd_record");
        ++calls;if(reset)++resets;
    }
};
