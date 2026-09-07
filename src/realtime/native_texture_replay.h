#pragma once
// Synthetic external producer: a separate queue owns shared NV12 textures and
// hands them to the exact same core API as a capture plugin. Input is a decoded
// fixture, not WGC/OBS. Copies/fences and real ownership are exercised explicitly.
struct NativeTextureReplay {
    struct Slot {
        ComPtr<ID3D11Texture2D> nv12;
        ComPtr<ID3D12Resource> opened;
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        UINT64 last_fence=0;
    };
    std::vector<Slot> slots;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    UINT64 next_fence=0,copy_count=0;
    bool init(ID3D11Device* device11,ID3D12Device* device12,UINT width,UINT height,UINT count,std::string& error) {
        D3D12_COMMAND_QUEUE_DESC q{};q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        if(!chain_check(device12->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)),"replay_queue",error) ||
           !chain_check(device12->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"replay_fence",error)) return false;
        slots.resize(count);
        for(auto& s:slots) {
            D3D11_TEXTURE2D_DESC d{};d.Width=width;d.Height=height;d.MipLevels=1;d.ArraySize=1;
            d.Format=DXGI_FORMAT_NV12;d.SampleDesc.Count=1;d.Usage=D3D11_USAGE_DEFAULT;
            d.BindFlags=D3D11_BIND_SHADER_RESOURCE;d.MiscFlags=D3D11_RESOURCE_MISC_SHARED;
            if(!chain_check(device11->CreateTexture2D(&d,nullptr,&s.nv12),"replay_shared_nv12",error))return false;
            ComPtr<IDXGIResource> shared;HANDLE handle=nullptr;
            if(!chain_check(s.nv12.As(&shared),"replay_dxgi",error) ||
               !chain_check(shared->GetSharedHandle(&handle),"replay_legacy_handle",error) ||
               !chain_check(device12->OpenSharedHandle(handle,IID_PPV_ARGS(&s.opened)),"replay_open12",error) ||
               !chain_check(device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&s.allocator)),"replay_allocator",error) ||
               !chain_check(device12->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,s.allocator.Get(),nullptr,IID_PPV_ARGS(&s.list)),"replay_list",error))return false;
            s.list->Close();
        }
        return true;
    }
    bool produce(UINT index, xess_gpu::ExternalGpuFrame& frame,std::string& error) {
        auto& s=slots[index];
        if(s.last_fence && !wait_queue_fence(fence.Get(),s.last_fence,30000)) {error="replay_slot_fence";return false;}
        if(FAILED(s.allocator->Reset()) || FAILED(s.list->Reset(s.allocator.Get(),nullptr))) {error="replay_slot_reset";return false;}
        auto* source=frame.frame.color.resource.Get();
        transition(s.list.Get(),source,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(s.list.Get(),s.opened.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
        s.list->CopyResource(s.opened.Get(),source);
        transition(s.list.Get(),source,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
        transition(s.list.Get(),s.opened.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
        if(FAILED(s.list->Close())) {error="replay_close";return false;}
        ID3D12CommandList* lists[]={s.list.Get()};queue->ExecuteCommandLists(1,lists);
        s.last_fence=++next_fence;
        if(FAILED(queue->Signal(fence.Get(),s.last_fence))) {error="replay_signal";return false;}
        frame.frame.color.resource=s.opened;
        frame.nv12_d3d11=s.nv12;
        // This producer uses independently allocated persistent textures rather
        // than a decoder pool. The owning texture pins that allocation itself.
        frame.frame.producer_pool_owner=s.nv12;
        frame.frame.producer_queue=queue;
        frame.frame.produced={fence,s.last_fence};
        ++copy_count;return true;
    }
};
