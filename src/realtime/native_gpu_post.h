#pragma once
struct NativeGpuPost {
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12QueryHeap> queries;ComPtr<ID3D12Resource> readback;
    UINT64 frequency=0;double gpu_seconds=0;
    UINT stride=0;uint64_t dispatch_count=0;
    bool init(ID3D12Device* device,ID3D12CommandQueue* queue,ID3D12RootSignature* root,const std::string& shaders,UINT slots,std::string& error) {
        D3D12_QUERY_HEAP_DESC q{};q.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;q.Count=slots*2;
        if(FAILED(queue->GetTimestampFrequency(&frequency)) || FAILED(device->CreateQueryHeap(&q,IID_PPV_ARGS(&queries)))) {error="post_timestamps";return false;}
        readback=create_buffer(device,slots*16,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,L"post GPU timestamps");
        if(!readback){error="post_timestamp_readback";return false;}
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors=slots*8;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        stride=device->GetDescriptorHandleIncrementSize(hd.Type);
        return chain_check(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)),"post_heap",error) &&
            chain_pipeline(device,shaders,"native_terminal_post",root,pso,error);
    }
    void record(ID3D12Device* device,ID3D12RootSignature* root,ID3D12GraphicsCommandList* list,UINT slot,
                ID3D12Resource* input,D3D12_RESOURCE_STATES input_state,ID3D12Resource* output) {
        const UINT b=slot*8;const auto desc=input->GetDesc();
        chain_srv(device,heap.Get(),b,stride,input,DXGI_FORMAT_R8G8B8A8_UNORM);
        for(UINT i=1;i<5;++i) chain_srv(device,heap.Get(),b+i,stride,nullptr,DXGI_FORMAT_R32_FLOAT);
        chain_uav(device,heap.Get(),b+5,stride,output,DXGI_FORMAT_R8G8B8A8_UNORM);
        chain_uav(device,heap.Get(),b+6,stride,nullptr,DXGI_FORMAT_R32_FLOAT);
        chain_transition(list,input,input_state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        chain_transition(list,output,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[]={heap.Get()};list->SetDescriptorHeaps(1,heaps);
        list->SetComputeRootSignature(root);list->SetPipelineState(pso.Get());
        list->SetComputeRootDescriptorTable(0,chain_gpu_desc(heap.Get(),b,stride));
        list->SetComputeRootDescriptorTable(1,chain_gpu_desc(heap.Get(),b+5,stride));
        UINT c[8]={static_cast<UINT>(desc.Width),desc.Height,25,0,0,0,0,0};
        list->SetComputeRoot32BitConstants(2,8,c,0);
        list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,slot*2);
        list->Dispatch((c[0]+7)/8,(c[1]+7)/8,1);
        list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,slot*2+1);
        list->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,slot*2,2,readback.Get(),slot*16);
        chain_transition(list,output,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);
        chain_transition(list,input,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,input_state);
        ++dispatch_count;
    }
    bool collect(UINT slot) {
        void* data=nullptr;D3D12_RANGE range{slot*16,slot*16+16};
        if(FAILED(readback->Map(0,&range,&data)))return false;
        const UINT64* times=static_cast<const UINT64*>(data)+slot*2;
        const bool valid=times[1]>=times[0];
        if(valid)gpu_seconds+=double(times[1]-times[0])/frequency;
        D3D12_RANGE written{0,0};readback->Unmap(0,&written);return valid;
    }
};
