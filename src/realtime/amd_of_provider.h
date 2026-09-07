#pragma once
#include "gpu_frame_contract.h"
#include "amd_of_context.h"

struct AmdProviderStatistics {
    uint64_t frames=0,pairs=0,sdk_calls=0,forward_calls=0,reverse_calls=0;
    uint64_t first_self_calls=0,dense_dispatches=0,history_resets=0,warmup_pairs=0;
    uint64_t reused_slots=0,full_pixel_cpu_bytes=0;
    std::string error;
};

// Included after the frozen native core so its small D3D helpers are shared.
// No decode/depth/SR/FG/post/encode implementation is duplicated here.
class AmdMotionProvider final:public xess_gpu::MotionProvider {
    struct Slot {
        ComPtr<ID3D12Resource> forward,reverse;
        D3D12_RESOURCE_STATES state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        xess_gpu::FencePoint retirement;
        bool used=false;
    };
    std::string shaders_;
    AmdProviderStatistics& stats_;
    xess_gpu::ProviderConfig config_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12DescriptorHeap> heap_;
    ComPtr<ID3D12PipelineState> dense_;
    UINT stride_=0;
    std::unique_ptr<AmdOfContext> forward_,reverse_;
    std::vector<Slot> slots_;
    bool reverse_started_=false;
    uint64_t forward_since_reset_=0,reverse_since_reset_=0,last_source_=xess_gpu::kNoFrame;
    uint64_t last_reset_source_=xess_gpu::kNoFrame;

    bool readable(const xess_gpu::TextureView* view) const {
        if(!view||!view->resource||view->format!=DXGI_FORMAT_R8G8B8A8_UNORM||
            view->state!=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE||view->valid.x||view->valid.y||
            view->valid.width!=config_.source.width||view->valid.height!=config_.source.height)return false;
        const auto desc=view->resource->GetDesc();
        return desc.Format==view->format&&desc.Width>=view->valid.width&&desc.Height>=view->valid.height;
    }
    xess_gpu::TextureView view(const ComPtr<ID3D12Resource>& resource) const {
        return {resource,DXGI_FORMAT_R32G32_FLOAT,config_.source,
            {0,0,config_.source.width,config_.source.height},D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    }
public:
    AmdMotionProvider(std::string shaders,AmdProviderStatistics& stats):shaders_(std::move(shaders)),stats_(stats){}
    const char* backend_name() const noexcept override{return "amd-of";}
    bool initialize(ID3D12Device* device,const xess_gpu::ProviderConfig& config,std::string& error) override {
        try {
            if(!device||!config.source.width||!config.source.height||config.slots<2||config.slots>8||!config.bidirectional)
                throw std::runtime_error("amd_provider_invalid_configuration");
            device_=device;config_=config;
            forward_=std::make_unique<AmdOfContext>(device,config.source.width,config.source.height);
            reverse_=std::make_unique<AmdOfContext>(device,config.source.width,config.source.height);
            D3D12_DESCRIPTOR_RANGE ranges[2]{};
            ranges[0].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV;ranges[0].NumDescriptors=1;
            ranges[1].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV;ranges[1].NumDescriptors=1;
            D3D12_ROOT_PARAMETER parameters[3]{};
            for(UINT i=0;i<2;++i){parameters[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                parameters[i].DescriptorTable.NumDescriptorRanges=1;parameters[i].DescriptorTable.pDescriptorRanges=&ranges[i];}
            parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            parameters[2].Constants.Num32BitValues=4;parameters[2].Constants.ShaderRegister=0;
            D3D12_ROOT_SIGNATURE_DESC rd{};rd.NumParameters=3;rd.pParameters=parameters;
            ComPtr<ID3DBlob> blob,errors;
            if(FAILED(D3D12SerializeRootSignature(&rd,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&errors))||
                FAILED(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root_))))
                throw std::runtime_error("amd_dense_root_signature");
            if(!chain_pipeline(device,shaders_,"native_amd_dense",root_.Get(),dense_,error))return false;
            D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            hd.NumDescriptors=4*config.slots;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            if(FAILED(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap_))))throw std::runtime_error("amd_dense_heap");
            stride_=device->GetDescriptorHandleIncrementSize(hd.Type);slots_.resize(config.slots);
            for(UINT i=0;i<config.slots;++i){
                auto& slot=slots_[i];
                slot.forward=AmdOfContext::texture(device,config.source.width,config.source.height,DXGI_FORMAT_R32G32_FLOAT);
                slot.reverse=AmdOfContext::texture(device,config.source.width,config.source.height,DXGI_FORMAT_R32G32_FLOAT);
                chain_srv(device,heap_.Get(),i*4,stride_,forward_->flow.Get(),DXGI_FORMAT_R16G16_SINT);
                chain_uav(device,heap_.Get(),i*4+1,stride_,slot.forward.Get(),DXGI_FORMAT_R32G32_FLOAT);
                chain_srv(device,heap_.Get(),i*4+2,stride_,reverse_->flow.Get(),DXGI_FORMAT_R16G16_SINT);
                chain_uav(device,heap_.Get(),i*4+3,stride_,slot.reverse.Get(),DXGI_FORMAT_R32G32_FLOAT);
            }
            return true;
        }catch(const std::exception& exception){error=exception.what();stats_.error=error;return false;}
    }

    bool record(const xess_gpu::FrameLease* previous,const xess_gpu::FrameLease& current,
                uint32_t index,xess_gpu::RecordContext& context,xess_gpu::MotionPacket& packet,
                xess_gpu::Counters& counters,std::string& error) override {
        using namespace xess_gpu;
        try{
            if(index>=slots_.size()||context.device!=device_.Get()||!context.list||!context.queue||
                !context.completion.fence||!context.completion.value||
                !same_adapter(current.metadata.adapter_luid,device_->GetAdapterLuid())||!readable(context.current_rgba))
                throw std::runtime_error("amd_provider_invalid_record_context_or_rgba");
            if(queue_&&queue_.Get()!=context.queue)throw std::runtime_error("amd_provider_history_queue_changed");
            if(!queue_)queue_=context.queue;
            auto& slot=slots_[index];
            if(slot.used&&!slot.retirement.complete())throw std::runtime_error("amd_provider_slot_not_retired");
            if(slot.used)++stats_.reused_slots;
            const bool first=previous==nullptr;
            if((first&&last_source_!=kNoFrame)||(!first&&
                (!readable(context.previous_rgba)||previous->metadata.source_frame_id!=current.metadata.previous_source_frame_id||
                 previous->metadata.source_frame_id!=last_source_)))throw std::runtime_error("amd_provider_source_pair_identity");
            bool reset=first||current.metadata.reset!=ResetReason::None;
            // A GPU scene decision is made after this provider record. Its
            // previous FrameLease conveys that decision at the next source.
            // The cut frame itself already receives core SR/FG/depth reset.
            if(previous&&previous->metadata.reset==ResetReason::SceneCut&&previous->metadata.source_frame_id!=last_reset_source_){
                reset=true;last_reset_source_=previous->metadata.source_frame_id;
            }
            if(reset){++stats_.history_resets;forward_since_reset_=0;reverse_since_reset_=0;
                if(current.metadata.reset!=ResetReason::None)last_reset_source_=current.metadata.source_frame_id;}
            forward_->record(context.list,context.current_rgba->resource.Get(),reset);
            ++forward_since_reset_;++stats_.forward_calls;++stats_.sdk_calls;++counters.directional_dispatch_count;
            packet={};packet.current_source_frame_id=current.metadata.source_frame_id;
            packet.previous_source_frame_id=current.metadata.previous_source_frame_id;
            packet.current_metadata=current.metadata;packet.produced=context.completion;
            packet.confidence_definition=ConfidenceDefinition::Unavailable;
            packet.confidence_semantics="AMD SDK has no confidence plane; common core computes actual cycle/photometric reliability. Official six-call warmup is retained.";
            if(first){
                ++stats_.first_self_calls;++counters.first_frame_self_analysis_count;packet.first_frame_self=true;
            }else{
                reverse_->record(context.list,context.current_rgba->resource.Get(),reset||!reverse_started_);
                reverse_->record(context.list,context.previous_rgba->resource.Get(),false);
                reverse_started_=true;reverse_since_reset_+=2;
                stats_.reverse_calls+=2;stats_.sdk_calls+=2;counters.directional_dispatch_count+=2;
                ++stats_.pairs;++counters.source_pair_analysis_count;
                if(forward_since_reset_<=6||reverse_since_reset_<=6)++stats_.warmup_pairs;
                chain_transition(context.list,forward_->flow.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chain_transition(context.list,reverse_->flow.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chain_transition(context.list,slot.forward.Get(),slot.state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chain_transition(context.list,slot.reverse.Get(),slot.state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                ID3D12DescriptorHeap* heaps[]={heap_.Get()};context.list->SetDescriptorHeaps(1,heaps);
                context.list->SetComputeRootSignature(root_.Get());context.list->SetPipelineState(dense_.Get());
                UINT values[4]={config_.source.width,config_.source.height,(config_.source.width+7)/8,(config_.source.height+7)/8};
                context.list->SetComputeRoot32BitConstants(2,4,values,0);
                for(UINT direction=0;direction<2;++direction){
                    const UINT base=index*4+direction*2;
                    context.list->SetComputeRootDescriptorTable(0,chain_gpu_desc(heap_.Get(),base,stride_));
                    context.list->SetComputeRootDescriptorTable(1,chain_gpu_desc(heap_.Get(),base+1,stride_));
                    context.list->Dispatch((values[0]+7)/8,(values[1]+7)/8,1);
                    ++stats_.dense_dispatches;++counters.motion_geometry_dispatch_count;
                }
                chain_transition(context.list,forward_->flow.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chain_transition(context.list,reverse_->flow.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chain_transition(context.list,slot.forward.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chain_transition(context.list,slot.reverse.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                slot.state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                packet.current_to_previous=view(slot.forward);packet.previous_to_current=view(slot.reverse);packet.has_reverse=true;
            }
            slot.retirement=context.completion;slot.used=true;last_source_=current.metadata.source_frame_id;++stats_.frames;
            return true;
        }catch(const std::exception& exception){error=exception.what();stats_.error=error;return false;}
    }
};
