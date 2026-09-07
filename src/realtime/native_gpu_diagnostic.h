#pragma once
// Explicit validation-only readback. Never selected by the normal core path.
// The caller waits its analysis fence first; every byte is counted separately.
inline bool native_dump_resource(Runtime& runtime, ID3D12Resource* resource,
                                 D3D12_RESOURCE_STATES state, const std::string& path,
                                 uint64_t& readback_bytes, std::string& error) {
    const auto desc=resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};UINT rows=0;UINT64 row_bytes=0,total=0;
    const bool buffer=desc.Dimension==D3D12_RESOURCE_DIMENSION_BUFFER;
    if(buffer) {rows=1;row_bytes=total=desc.Width;}
    else runtime.device->GetCopyableFootprints(&desc,0,1,0,&footprint,&rows,&row_bytes,&total);
    auto readback=create_buffer(runtime.device.Get(),total,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,L"explicit core diagnostic");
    ComPtr<ID3D12CommandAllocator> allocator;ComPtr<ID3D12GraphicsCommandList> list;
    if(!readback || FAILED(runtime.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator))) ||
       FAILED(runtime.device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)))) {
        error="diagnostic_resources";return false;
    }
    transition(list.Get(),resource,state,D3D12_RESOURCE_STATE_COPY_SOURCE);
    if(buffer) list->CopyBufferRegion(readback.Get(),0,resource,0,total);
    else {
        D3D12_TEXTURE_COPY_LOCATION src{};src.pResource=resource;src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION dst{};dst.pResource=readback.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=footprint;
        list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    }
    transition(list.Get(),resource,D3D12_RESOURCE_STATE_COPY_SOURCE,state);
    if(FAILED(list->Close())) {error="diagnostic_close";return false;}
    ID3D12CommandList* lists[]={list.Get()};runtime.queue->ExecuteCommandLists(1,lists);UINT64 fence=0;
    if(!runtime.signal_queue(&fence) || !runtime.wait_fence(fence,30000)) {error="diagnostic_fence";return false;}
    void* data=nullptr;D3D12_RANGE range{0,static_cast<SIZE_T>(total)};
    if(FAILED(readback->Map(0,&range,&data))) {error="diagnostic_map";return false;}
    std::ofstream output(path,std::ios::binary|std::ios::trunc);
    for(UINT y=0;y<rows;++y) output.write(static_cast<const char*>(data)+(buffer?0:footprint.Offset+static_cast<size_t>(y)*footprint.Footprint.RowPitch),row_bytes);
    D3D12_RANGE written{0,0};readback->Unmap(0,&written);
    if(!output) {error="diagnostic_write";return false;}
    readback_bytes+=row_bytes*rows;return true;
}
