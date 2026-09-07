#pragma once
struct NativeGpuTimers {
    ComPtr<ID3D12QueryHeap> queries;
    ComPtr<ID3D12Resource> readback;
    UINT64 frequency=0,frames=0;
    double motion_seconds=0,mask_seconds=0,depth_post_seconds=0,sr_seconds=0;
    bool init(ID3D12Device* device,ID3D12CommandQueue* queue,UINT slots) {
        D3D12_QUERY_HEAP_DESC d{};d.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;d.Count=slots*8;
        if(FAILED(queue->GetTimestampFrequency(&frequency)) || !frequency ||
           FAILED(device->CreateQueryHeap(&d,IID_PPV_ARGS(&queries)))) return false;
        readback=create_buffer(device,slots*8*sizeof(UINT64),D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,L"GPU stage timestamps");
        return readback!=nullptr;
    }
    void mark(ID3D12GraphicsCommandList* list,UINT slot,UINT point) {list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,slot*8+point);}
    void resolve(ID3D12GraphicsCommandList* list,UINT slot) {list->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,slot*8,7,readback.Get(),slot*8*sizeof(UINT64));}
    bool collect(UINT slot,bool sr) {
        void* data=nullptr;D3D12_RANGE range{slot*8*sizeof(UINT64),(slot*8+7)*sizeof(UINT64)};
        if(FAILED(readback->Map(0,&range,&data)))return false;
        const UINT64* t=static_cast<const UINT64*>(data)+slot*8;
        bool valid=true;for(UINT i=1;i<7;++i) valid=valid&&t[i]>=t[i-1];
        if(valid) {motion_seconds+=double(t[1]-t[0])/frequency;mask_seconds+=double(t[2]-t[1])/frequency;
            depth_post_seconds+=double(t[3]-t[2])/frequency;if(sr)sr_seconds+=double(t[6]-t[5])/frequency;++frames;}
        D3D12_RANGE written{0,0};readback->Unmap(0,&written);return valid;
    }
};
