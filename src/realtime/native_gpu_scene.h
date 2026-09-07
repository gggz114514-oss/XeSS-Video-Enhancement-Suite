#pragma once
// Sixteen-byte, per-slot scene metadata. The SDK reset arguments live on the
// CPU, so the host waits this submission's fence before recording consumers.
struct NativeGpuScene {
    struct Slot {GpuResource result; ComPtr<ID3D12Resource> readback;};
    std::vector<Slot> slots;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12PipelineState> pso;
    UINT stride=0;
    bool init(ID3D12Device* device, ID3D12RootSignature* root, const std::string& shaders, UINT count, std::string& error) {
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors=count*8;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (!chain_check(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)),"scene_heap",error) ||
            !chain_pipeline(device,shaders,"native_scene_stats",root,pso,error)) return false;
        stride=device->GetDescriptorHandleIncrementSize(hd.Type);slots.resize(count);
        for(auto& s:slots) {
            D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=16;d.Height=1;
            d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            D3D12_HEAP_PROPERTIES p{};p.Type=D3D12_HEAP_TYPE_DEFAULT;
            s.result.state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            if(!chain_check(device->CreateCommittedResource(&p,D3D12_HEAP_FLAG_NONE,&d,s.result.state,nullptr,
                IID_PPV_ARGS(&s.result.resource)),"scene_result",error)) return false;
            s.readback=create_buffer(device,16,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,L"scene metadata16bytes");
            if(!s.readback) {error="scene_readback";return false;}
        }
        return true;
    }
    void record(ID3D12Device* device,ID3D12RootSignature* root,ID3D12GraphicsCommandList* list,UINT slot,
                ID3D12Resource* previous,ID3D12Resource* current,ID3D12Resource* mask,UINT w,UINT h) {
        const UINT b=slot*8;auto& s=slots[slot];
        chain_srv(device,heap.Get(),b,stride,previous,DXGI_FORMAT_R32_FLOAT);
        chain_srv(device,heap.Get(),b+1,stride,current,DXGI_FORMAT_R32_FLOAT);
        chain_srv(device,heap.Get(),b+2,stride,mask,DXGI_FORMAT_R8_UNORM);
        for(UINT i=3;i<5;++i) chain_srv(device,heap.Get(),b+i,stride,nullptr,DXGI_FORMAT_R32_FLOAT);
        D3D12_UNORDERED_ACCESS_VIEW_DESC v{};v.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;
        v.Buffer.NumElements=4;v.Buffer.StructureByteStride=4;
        device->CreateUnorderedAccessView(s.result.resource.Get(),nullptr,&v,chain_cpu_desc(heap.Get(),b+5,stride));
        chain_uav(device,heap.Get(),b+6,stride,nullptr,DXGI_FORMAT_R32_FLOAT);
        chain_transition(list,s.result.resource.Get(),s.result.state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[]={heap.Get()};list->SetDescriptorHeaps(1,heaps);
        list->SetComputeRootSignature(root);list->SetPipelineState(pso.Get());
        list->SetComputeRootDescriptorTable(0,chain_gpu_desc(heap.Get(),b,stride));
        list->SetComputeRootDescriptorTable(1,chain_gpu_desc(heap.Get(),b+5,stride));
        UINT c[8]={w,h,0,0,0,0,0,0};list->SetComputeRoot32BitConstants(2,8,c,0);list->Dispatch(1,1,1);
        chain_transition(list,s.result.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        s.result.state=D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->CopyBufferRegion(s.readback.Get(),0,s.result.resource.Get(),0,16);
    }
    bool read(UINT index, float (&values)[4]) {
        void* data=nullptr;D3D12_RANGE range{0,16};
        if(FAILED(slots[index].readback->Map(0,&range,&data))) return false;
        std::memcpy(values,data,16);D3D12_RANGE written{0,0};slots[index].readback->Unmap(0,&written);
        return std::isfinite(values[0]) && std::isfinite(values[1]) && std::isfinite(values[2]);
    }
    static bool cut(const float (&s)[4]) {
        // A flat-color cut can retain perfect roundtrip flow while essentially
        // all histogram mass changes. Do not let that vacuous reliability veto
        // an extreme photometric discontinuity (hard-cut fixture source4).
        return (s[0]>52 && s[2]<.30f) || (s[1]>.62f && s[2]<.45f) ||
            (s[0]>75 && s[1]>.48f) || (s[1]>.95f && s[0]>30);
    }
};
