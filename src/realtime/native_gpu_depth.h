#pragma once
#include "quality_depth_openvino.h"
#include "gpu_frame_contract.h"
#include "native_depth_rgb_preprocess.h"

// Uses the chain's texture/descriptor helpers. D3D11 legacy shared buffers are
// copied to a shader-readable D3D12 buffer; this is an explicit GPU copy.
struct NativeGpuDepth {
    struct Slot {
        ComPtr<ID3D11Buffer> output11;
        ComPtr<ID3D12Resource> output12;
        GpuResource raw, limits, normalized, stable, resized;
        GpuResource summary;
        ComPtr<ID3D12Resource> summary_readback;
    };
    xess_full_gpu_quality::CachedDepthAnything model;
    NativeDepthRgbPreprocess rgb_preprocess;
    bool rgb_input=true;
    std::vector<Slot> slots;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12PipelineState> reduce, normalize, temporal, resize, summarize;
    ID3D12Device* device = nullptr;
    UINT stride = 0, width = 0, height = 0, out_width = 0, out_height = 0;
    uint64_t inference_count = 0, gpu_copy_count = 0, pack_count = 0;
    double inference_seconds = 0;
    std::vector<std::vector<float>> observed_statistics;
    uint64_t statistics_readback_bytes = 0;
    GpuResource descriptor_sentinel;
    float sentinel_value=0;

    bool buffer(UINT bytes, D3D12_RESOURCE_FLAGS flags, GpuResource& value,
                std::string& error) {
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = bytes; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
        d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        d.Flags = flags;
        D3D12_HEAP_PROPERTIES p{}; p.Type = D3D12_HEAP_TYPE_DEFAULT;
        value.state = D3D12_RESOURCE_STATE_COMMON;
        return chain_check(device->CreateCommittedResource(&p, D3D12_HEAP_FLAG_NONE,
            &d, value.state, nullptr, IID_PPV_ARGS(&value.resource)), "depth_buffer", error);
    }
    bool init(ID3D11Device* d11, ID3D12Device* d12, UINT w, UINT h, UINT ow,
              UINT oh, UINT count, const std::string& path,
              const std::string& shaders, std::string& error,
              UINT allocation_w=0,UINT allocation_h=0,bool use_rgb_input=true) {
        device = d12; width = w; height = h; out_width = ow; out_height = oh;
        rgb_input=use_rgb_input;
        if(const char* value=std::getenv("XESS_GPU_TEST_DESCRIPTOR_SENTINEL")) {
            sentinel_value=static_cast<float>(std::atof(value));
            if(!std::isfinite(sentinel_value) || !chain_texture(device,1,1,DXGI_FORMAT_R32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,descriptor_sentinel,error))return false;
        }
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 0, 0, 0};
        ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 0};
        D3D12_ROOT_PARAMETER parameters[3]{};
        for (UINT i = 0; i < 2; ++i) {
            parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[i].DescriptorTable = {1, &ranges[i]};
        }
        parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[2].Constants = {0, 0, 8};
        D3D12_ROOT_SIGNATURE_DESC rd{3, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob;
        if (!chain_check(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1,
                 &blob, nullptr), "depth_root_serialize", error) ||
            !chain_check(device->CreateRootSignature(0, blob->GetBufferPointer(),
                 blob->GetBufferSize(), IID_PPV_ARGS(&root)), "depth_root", error)) return false;
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = count * 40; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (!chain_check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "depth_heap", error)) return false;
        stride = device->GetDescriptorHandleIncrementSize(hd.Type);
        if (!chain_pipeline(device, shaders, "native_depth_reduce", root.Get(), reduce, error) ||
            !chain_pipeline(device, shaders, "native_depth_normalize", root.Get(), normalize, error) ||
            !chain_pipeline(device, shaders, "native_depth_temporal", root.Get(), temporal, error) ||
            !chain_pipeline(device, shaders, "native_depth_resize", root.Get(), resize, error) ||
            !chain_pipeline(device, shaders, "native_depth_summary", root.Get(), summarize, error)) return false;
        slots.resize(count);
        for (auto& s : slots) {
            D3D11_BUFFER_DESC bd{}; bd.ByteWidth = 518 * 518 * sizeof(float);
            bd.Usage = D3D11_USAGE_DEFAULT;
            bd.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
            // The previously proven OpenVINO DX_BUFFER allocation: bind=0.
            if (!chain_check(d11->CreateBuffer(&bd, nullptr, &s.output11), "depth_remote_buffer11", error)) return false;
            ComPtr<IDXGIResource> shared; HANDLE handle = nullptr;
            if (!chain_check(s.output11.As(&shared), "depth_dxgi", error) ||
                !chain_check(shared->GetSharedHandle(&handle), "depth_legacy_handle", error) ||
                !chain_check(device->OpenSharedHandle(handle, IID_PPV_ARGS(&s.output12)), "depth_open12", error)) return false;
            // Legacy handles are not NT handles and MUST NOT be CloseHandle'd.
            const auto desc = s.output12->GetDesc();
            if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || desc.Width < bd.ByteWidth) {
                error = "depth_shared_buffer_layout"; return false;
            }
            if (!buffer(bd.ByteWidth, D3D12_RESOURCE_FLAG_NONE, s.raw, error) ||
                !buffer(16, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, s.limits, error) ||
                !buffer(64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, s.summary, error) ||
                !chain_texture(device, w, h, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, s.normalized, error) ||
                !chain_texture(device, w, h, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, s.stable, error) ||
                !chain_texture(device, ow, oh, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, s.resized, error)) return false;
            s.summary_readback = create_buffer(device,64,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,L"depth statistics 64 bytes");
            if (!s.summary_readback) {error="depth_statistics_readback";return false;}
        }
        if(rgb_input)return rgb_preprocess.init(d11,w,h,allocation_w?allocation_w:w,allocation_h?allocation_h:h,count,shaders,error)&&
            model.initialize_preprocessed(path,d11,error);
        return model.initialize(path, d11, "GPU", w, h, error);
    }
    bool infer(UINT index, ID3D11Texture2D* nv12,const xess_gpu::FrameMetadata& metadata,
               const xess_gpu::Crop& crop,std::string& error) {
        const auto started = std::chrono::steady_clock::now();
        if(metadata.range!=xess_gpu::Range::Limited||(metadata.matrix!=xess_gpu::Matrix::Bt601&&metadata.matrix!=xess_gpu::Matrix::Bt709)||
           metadata.transfer==xess_gpu::Transfer::Pq||metadata.transfer==xess_gpu::Transfer::Hlg) {error="depth_unsupported_color_profile";return false;}
        if(rgb_input) {
            if(!rgb_preprocess.run(index,nv12,metadata,crop,error)||
               !model.infer_preprocessed(rgb_preprocess.slots[index].remote.Get(),slots[index].output11.Get(),error))return false;
        } else if (!model.infer(nv12, slots[index].output11.Get(), error)) return false;
        inference_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        ++inference_count;
        return true;
    }
    void transition_resource(ID3D12GraphicsCommandList* list, GpuResource& r,
                             D3D12_RESOURCE_STATES to) {
        chain_transition(list, r.resource.Get(), r.state, to); r.state = to;
    }
    void buffer_srv(UINT index, GpuResource& r, UINT count) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.NumElements = count; d.Buffer.StructureByteStride = sizeof(float);
        device->CreateShaderResourceView(r.resource.Get(), &d, chain_cpu_desc(heap.Get(), index, stride));
    }
    void record(UINT index, int previous_index, ID3D12GraphicsCommandList* list,
                ID3D12Resource* current_to_previous, ID3D12Resource* confidence,
                ID3D12Resource* mask, bool reset) {
        auto& s = slots[index]; const UINT b = index * 40;
        chain_transition(list, s.output12.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition_resource(list, s.raw, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(s.raw.resource.Get(), 0, s.output12.Get(), 0, 518 * 518 * sizeof(float));
        chain_transition(list, s.output12.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        transition_resource(list, s.raw, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ++gpu_copy_count;
        ID3D12DescriptorHeap* heaps[] = {heap.Get()}; list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(root.Get());
        auto bind = [&](UINT offset, ID3D12PipelineState* pso, const UINT* constants, UINT w, UINT h) {
            list->SetPipelineState(pso);
            list->SetComputeRootDescriptorTable(0, chain_gpu_desc(heap.Get(), b + offset, stride));
            list->SetComputeRootDescriptorTable(1, chain_gpu_desc(heap.Get(), b + offset + 5, stride));
            list->SetComputeRoot32BitConstants(2, 8, constants, 0);
            list->Dispatch(w, h, 1);
        };
        // Every recorded dispatch has its own stable descriptor range. Unused
        // neighbours are null views, never aliased outputs from another pass.
        for (UINT i = 0; i < 40; ++i) chain_srv(device, heap.Get(), b+i, stride, nullptr, DXGI_FORMAT_R32_FLOAT);
        if(descriptor_sentinel.resource) {
            // Poison each pass's unused second UAV and immediately adjacent
            // descriptor. Neither may influence the actual bound depth planes.
            for(UINT offset=0;offset<40;offset+=8) {
                chain_uav(device,heap.Get(),b+offset+6,stride,descriptor_sentinel.resource.Get(),DXGI_FORMAT_R32_FLOAT);
                chain_srv(device,heap.Get(),b+offset+7,stride,descriptor_sentinel.resource.Get(),DXGI_FORMAT_R32_FLOAT);
            }
            const float v[4]={sentinel_value,sentinel_value,sentinel_value,sentinel_value};
            list->ClearUnorderedAccessViewFloat(chain_gpu_desc(heap.Get(),b+6,stride),chain_cpu_desc(heap.Get(),b+6,stride),descriptor_sentinel.resource.Get(),v,0,nullptr);
        }
        buffer_srv(b, s.raw, 518 * 518);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
        uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uv.Buffer.NumElements = 4; uv.Buffer.StructureByteStride = sizeof(float);
        device->CreateUnorderedAccessView(s.limits.resource.Get(), nullptr, &uv, chain_cpu_desc(heap.Get(), b+5, stride));
        transition_resource(list, s.limits, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        UINT c[8] = {width, height, 518, 518, 0, 0, 0, 0};
        bind(0, reduce.Get(), c, 1, 1);
        transition_resource(list, s.limits, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        buffer_srv(b+8, s.raw, 518*518); buffer_srv(b+9, s.limits, 4);
        chain_uav(device, heap.Get(), b+13, stride, s.normalized.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        transition_resource(list, s.normalized, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        bind(8, normalize.Get(), c, (width+7)/8, (height+7)/8);
        transition_resource(list, s.normalized, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ++pack_count;
        ID3D12Resource* prior = !reset && previous_index >= 0 ? slots[previous_index].stable.resource.Get() : s.normalized.resource.Get();
        chain_srv(device, heap.Get(), b+16, stride, s.normalized.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_srv(device, heap.Get(), b+17, stride, prior, DXGI_FORMAT_R32_FLOAT);
        chain_srv(device, heap.Get(), b+18, stride, current_to_previous, DXGI_FORMAT_R16G16_FLOAT);
        chain_srv(device, heap.Get(), b+19, stride, confidence, DXGI_FORMAT_R32_FLOAT);
        chain_srv(device, heap.Get(), b+20, stride, mask, DXGI_FORMAT_R8_UNORM);
        chain_uav(device, heap.Get(), b+21, stride, s.stable.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        transition_resource(list, s.stable, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        c[4] = reset ? 1 : 0;
        bind(16, temporal.Get(), c, (width+7)/8, (height+7)/8);
        transition_resource(list, s.stable, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        chain_srv(device, heap.Get(), b+24, stride, s.stable.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        chain_uav(device, heap.Get(), b+29, stride, s.resized.resource.Get(), DXGI_FORMAT_R32_FLOAT);
        transition_resource(list, s.resized, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        c[0] = out_width; c[1] = out_height; c[2] = width; c[3] = height;
        bind(24, resize.Get(), c, (out_width+7)/8, (out_height+7)/8);
        transition_resource(list, s.resized, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        buffer_srv(b+32,s.limits,4);
        chain_srv(device,heap.Get(),b+33,stride,s.stable.resource.Get(),DXGI_FORMAT_R32_FLOAT);
        chain_srv(device,heap.Get(),b+34,stride,s.resized.resource.Get(),DXGI_FORMAT_R32_FLOAT);
        chain_srv(device,heap.Get(),b+35,stride,mask,DXGI_FORMAT_R8_UNORM);
        uv.Buffer.NumElements=16;
        device->CreateUnorderedAccessView(s.summary.resource.Get(),nullptr,&uv,chain_cpu_desc(heap.Get(),b+37,stride));
        transition_resource(list,s.summary,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        c[0]=width;c[1]=height;c[2]=out_width;c[3]=out_height;
        bind(32,summarize.Get(),c,1,1);
        transition_resource(list,s.summary,D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(s.summary_readback.Get(),0,s.summary.resource.Get(),0,64);
    }
    // Called only after this slot's already-required reuse/drain fence. It never
    // inserts a queue wait or reads a pixel plane. NaN/constant remote output
    // fails closed; a flat input is not used as an acceptance fixture.
    bool validate_statistics(UINT index, std::string& error) {
        void* data=nullptr; D3D12_RANGE range{0,64};
        if (FAILED(slots[index].summary_readback->Map(0,&range,&data))) {error="depth_statistics_map";return false;}
        const float* values=static_cast<const float*>(data);
        std::vector<float> copy(values,values+16);
        D3D12_RANGE written{0,0};slots[index].summary_readback->Unmap(0,&written);
        statistics_readback_bytes+=64;
        if(observed_statistics.size()<4096) observed_statistics.push_back(copy);
        for(float value:copy) if(!std::isfinite(value)) {error="depth_statistics_nonfinite";return false;}
        if (!(copy[1]>copy[0] && copy[2]==518*518 && copy[3]==0 &&
              copy[4]>=0 && copy[5]<=1 && copy[8]>=0 && copy[9]<=1 && copy[14]==0)) {
            error="depth_statistics_invalid";return false;
        }
        return true;
    }
};
