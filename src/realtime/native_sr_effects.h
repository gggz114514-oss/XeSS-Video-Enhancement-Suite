#pragma once
// Included after FgFrameSlot. Independent descriptor ranges for every stage
// and in-flight slot: never rewrite a descriptor already referenced by a list.
struct NativeSrEffects {
    struct Scratch {ComPtr<ID3D12Resource> guide,mask,horizontal,result;};
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12DescriptorHeap> heap;
    std::vector<Scratch> scratch;
    UINT stride=0,history_count=0; bool guard=false,five=false;
    uint64_t frames=0,history_samples=0,resets=0;
    bool init(ID3D12Device* device,const std::string& shaders,UINT slots,UINT w,UINT h,
              bool use_guard,bool use_five,std::string& error) {
        guard=use_guard;five=use_five;
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,13,0,0,0};
        ranges[1]={D3D12_DESCRIPTOR_RANGE_TYPE_UAV,2,0,0,0};
        D3D12_ROOT_PARAMETER parameters[3]{};
        for(UINT i=0;i<2;++i){parameters[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;parameters[i].DescriptorTable={1,&ranges[i]};}
        parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;parameters[2].Constants={0,0,8};
        D3D12_ROOT_SIGNATURE_DESC desc{3,parameters,0,nullptr,D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob,errors;
        if(FAILED(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&errors)) ||
           FAILED(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)))) {error="sr_effects_root";return false;}
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors=slots*4*16;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        stride=device->GetDescriptorHandleIncrementSize(hd.Type);
        if(!chain_check(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)),"sr_effects_heap",error) ||
           !chain_pipeline(device,shaders,"native_sr_effects",root.Get(),pso,error))return false;
        scratch.resize(slots);
        for(auto& s:scratch) {
            auto make=[&](DXGI_FORMAT format){return create_texture(device,w,h,format,D3D12_RESOURCE_STATE_COMMON,L"SR effects scratch",D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);};
            s.result=make(DXGI_FORMAT_R8G8B8A8_UNORM);
            if(guard){s.guide=make(DXGI_FORMAT_R8G8B8A8_UNORM);s.mask=make(DXGI_FORMAT_R32_FLOAT);s.horizontal=make(DXGI_FORMAT_R32_FLOAT);}
            if(!s.result || (guard&&(!s.guide||!s.mask||!s.horizontal))){error="sr_effects_texture";return false;}
        }
        return true;
    }
    void record(ID3D12Device* device,ID3D12GraphicsCommandList* list,std::vector<FgFrameSlot>& slots,UINT index,bool reset) {
        if(reset){history_count=0;++resets;}
        auto& current=slots[index];auto& s=scratch[index];
        const UINT count=five?history_count:0;
        const UINT w=static_cast<UINT>(current.sr_output->GetDesc().Width),h=current.sr_output->GetDesc().Height;
        const UINT iw=static_cast<UINT>(current.color->GetDesc().Width),ih=current.color->GetDesc().Height;
        transition(list,current.sr_output.Get(),current.sr_state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        current.sr_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        for(UINT d=0;d<=count;++d) {
            auto& f=slots[(index+static_cast<UINT>(slots.size())-d)%slots.size()];
            transition(list,f.color.Get(),f.color_state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);f.color_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            if(d<count){transition(list,f.forward_flow.Get(),f.forward_state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);f.forward_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;}
        }
        ID3D12DescriptorHeap* heaps[]={heap.Get()};list->SetDescriptorHeaps(1,heaps);
        list->SetComputeRootSignature(root.Get());list->SetPipelineState(pso.Get());
        for(UINT pass=guard?0:3;pass<4;++pass) {
            const UINT b=(index*4+pass)*16;
            auto srv=[&](UINT n,ID3D12Resource* r,DXGI_FORMAT format){chain_srv(device,heap.Get(),b+n,stride,r,format);};
            srv(0,pass==3?current.sr_output.Get():nullptr,DXGI_FORMAT_R8G8B8A8_UNORM);
            srv(1,current.color.Get(),DXGI_FORMAT_R8G8B8A8_UNORM);
            srv(2,pass==1||pass==3?s.guide.Get():nullptr,DXGI_FORMAT_R8G8B8A8_UNORM);
            srv(3,pass==2?s.mask.Get():nullptr,DXGI_FORMAT_R32_FLOAT);
            srv(4,pass==3?s.horizontal.Get():nullptr,DXGI_FORMAT_R32_FLOAT);
            for(UINT d=0;d<4;++d){
                srv(5+d,pass==3&&d<count?slots[(index+slots.size()-d-1)%slots.size()].color.Get():nullptr,DXGI_FORMAT_R8G8B8A8_UNORM);
                // Contract: despite its legacy name, forward_flow is current->previous.
                srv(9+d,pass==3&&d<count?slots[(index+slots.size()-d)%slots.size()].forward_flow.Get():nullptr,DXGI_FORMAT_R16G16_FLOAT);
            }
            ID3D12Resource* output=pass==0?s.guide.Get():pass==1?s.mask.Get():pass==2?s.horizontal.Get():s.result.Get();
            chain_uav(device,heap.Get(),b+13,stride,pass==0||pass==3?output:nullptr,DXGI_FORMAT_R8G8B8A8_UNORM);
            chain_uav(device,heap.Get(),b+14,stride,pass==1||pass==2?output:nullptr,DXGI_FORMAT_R32_FLOAT);
            transition(list,output,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list->SetComputeRootDescriptorTable(0,chain_gpu_desc(heap.Get(),b,stride));
            list->SetComputeRootDescriptorTable(1,chain_gpu_desc(heap.Get(),b+13,stride));
            UINT c[8]={w,h,iw,ih,pass,guard?1u:0u,five?1u:0u,count};list->SetComputeRoot32BitConstants(2,8,c,0);
            list->Dispatch((w+7)/8,(h+7)/8,1);
            transition(list,output,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        // Preserve the consumer contract: FG and encoder both see the same
        // processed SR surface. One explicit GPU copy, never host staging.
        transition(list,s.result.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(list,current.sr_output.Get(),current.sr_state,D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyResource(current.sr_output.Get(),s.result.Get());current.sr_state=D3D12_RESOURCE_STATE_COPY_DEST;
        transition(list,s.result.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
        if(guard)for(auto* r:{s.guide.Get(),s.mask.Get(),s.horizontal.Get()})transition(list,r,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
        ++frames;history_samples+=count;history_count=std::min(4u,history_count+1);
    }
};
