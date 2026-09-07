#pragma once

// Per-slot OpenVINO Intel-GPU bridge for the independent Quality worker.
// Core/compiled model/infer request are built once; each infer only wraps the
// already-owned D3D11 decoder planes and shared FP32 output buffer.

#if __has_include(<openvino/openvino.hpp>) && \
    __has_include(<openvino/runtime/intel_gpu/ocl/dx.hpp>)

#define XESS_FG_QUALITY_HAS_OPENVINO 1
#include <d3d11.h>
#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <CL/cl_d3d11.h>
#include <wrl/client.h>

#include <memory>
#include <string>
#include <map>
#include <chrono>

namespace xess_full_gpu_quality {

class CachedDepthAnything {
public:
    ~CachedDepthAnything(){reset();}
    // Already normalized, GPU-resident NCHW input. Color/crop/resize semantics
    // are owned by the validated native preprocessing stage, not NV12toRGB.
    bool initialize_preprocessed(const std::string& model_path,ID3D11Device* device,std::string& error) {
        if(!device||model_path.empty()){error="depth_rgb_device_or_model_missing";return false;}
        try {
            core_=std::make_unique<ov::Core>();
            auto model=core_->read_model(model_path);model->reshape({1,3,518,518});
            context_=std::make_unique<ov::intel_gpu::ocl::D3DContext>(*core_,device);
            cl::Context cl_context(context_->get(),true);
            auto devices=cl_context.getInfo<CL_CONTEXT_DEVICES>();
            rgb_queue_=std::make_unique<cl::CommandQueue>(cl_context,devices.at(0),0);
            rgb_context_=std::make_unique<ov::intel_gpu::ocl::ClContext>(*core_,rgb_queue_->get());
            cl_platform_id platform=nullptr;
            if(clGetDeviceInfo(devices.at(0)(),CL_DEVICE_PLATFORM,sizeof(platform),&platform,nullptr)!=CL_SUCCESS)throw std::runtime_error("OpenCL platform query");
            const auto extensions=devices.at(0).getInfo<CL_DEVICE_EXTENSIONS>();
            acquire_=reinterpret_cast<clEnqueueAcquireD3D11ObjectsKHR_fn>(clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueAcquireD3D11ObjectsKHR"));
            release_=reinterpret_cast<clEnqueueReleaseD3D11ObjectsKHR_fn>(clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueReleaseD3D11ObjectsKHR"));
            if(extensions.find("cl_khr_d3d11_sharing")==std::string::npos||!acquire_||!release_)throw std::runtime_error("D3D11 sharing unavailable");
            compiled_=std::make_unique<ov::CompiledModel>(core_->compile_model(model,*rgb_context_,ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));
            request_=std::make_unique<ov::InferRequest>(compiled_->create_infer_request());
            return true;
        }catch(const std::exception& e){error=std::string("depth_rgb_init:")+e.what();reset();return false;}
    }
    bool infer_preprocessed(ID3D11Buffer* input,ID3D11Buffer* output,std::string& error) {
        if(!request_||!context_||!input||!output){error="depth_rgb_not_ready";return false;}
        try {
            auto it=rgb_inputs_.find(input);
            if(it==rgb_inputs_.end()) {
                RgbInput entry;
                entry.shared=context_->create_tensor(ov::element::f32,{1,3,518,518},input);
                auto shared_tensor=entry.shared.as<ov::intel_gpu::ocl::D3DBufferTensor>();
                entry.stable=rgb_context_->create_tensor(ov::element::f32,{1,3,518,518},shared_tensor.get());
                it=rgb_inputs_.emplace(input,std::move(entry)).first;
            }
            auto shared_tensor=it->second.shared.as<ov::intel_gpu::ocl::D3DBufferTensor>();
            auto output_it=rgb_outputs_.find(output);
            if(output_it==rgb_outputs_.end()) {
                RgbInput entry;entry.shared=context_->create_tensor(ov::element::f32,{1,518,518},output);
                auto output_shared=entry.shared.as<ov::intel_gpu::ocl::D3DBufferTensor>();
                entry.stable=rgb_context_->create_tensor(ov::element::f32,{1,518,518},output_shared.get());
                output_it=rgb_outputs_.emplace(output,std::move(entry)).first;
            }
            auto output_shared=output_it->second.shared.as<ov::intel_gpu::ocl::D3DBufferTensor>();
            cl_mem shared[2]={shared_tensor.get(),output_shared.get()};
            // Explicit acquire and inference share one in-order queue. Import
            // the CL handles in that engine as OCL_BUFFER, avoiding accidental
            // double acquire by the plugin. No extra pixel copy is needed.
            auto status=acquire_(rgb_queue_->get(),2,shared,0,nullptr,nullptr);
            if(status!=CL_SUCCESS)throw std::runtime_error("RGB acquire:"+std::to_string(status));
            ++rgb_acquire_count;
            try {
                request_->set_input_tensor(it->second.stable);request_->set_output_tensor(output_it->second.stable);
                request_->infer();
            } catch(...) {release_(rgb_queue_->get(),2,shared,0,nullptr,nullptr);clFlush(rgb_queue_->get());throw;}
            cl_event retired=nullptr;status=release_(rgb_queue_->get(),2,shared,0,nullptr,&retired);
            if(status!=CL_SUCCESS)throw std::runtime_error("RGB release:"+std::to_string(status));
            // Only the necessary ownership-transfer event, not clFinish or a
            // full D3D queue wait. D3D12 aliases may consume output afterwards.
            const auto wait_start=std::chrono::steady_clock::now();
            status=clWaitForEvents(1,&retired);clReleaseEvent(retired);++rgb_release_wait_count;
            rgb_release_wait_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-wait_start).count();
            if(status!=CL_SUCCESS)throw std::runtime_error("RGB release event:"+std::to_string(status));
            return true;
        }catch(const std::exception& e){error=std::string("depth_rgb_infer:")+e.what();return false;}
    }
    bool initialize(const std::string& model_path, ID3D11Device* device,
                    const std::string& device_name, size_t source_width,
                    size_t source_height, std::string& error) {
        if (!device) { error = "depth_d3d11_device_missing"; return false; }
        if (model_path.empty()) { error = "depth_model_missing"; return false; }
        if (!source_width || !source_height) {
            error = "depth_source_size_missing"; return false;
        }
        source_width_ = source_width;
        source_height_ = source_height;
        try {
            core_ = std::make_unique<ov::Core>();
            auto model = core_->read_model(model_path);
            model->reshape({1, 3, 518, 518});
            ov::preprocess::PrePostProcessor ppp(model);
            auto input = model->input();
            auto& tensor = ppp.input(input.get_any_name()).tensor();
            tensor.set_element_type(ov::element::u8)
                .set_spatial_static_shape(source_height, source_width)
                .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES,
                                  {"y", "uv"})
                .set_memory_type(ov::intel_gpu::memory_type::surface);
            // OpenVINO scale divides.  This is exactly RGB/255 followed by
            // ImageNet normalization, then model NCHW layout (never BGR).
            ppp.input(input.get_any_name()).preprocess()
                .convert_color(ov::preprocess::ColorFormat::RGB)
                .convert_element_type(ov::element::f32)
                .resize(ov::preprocess::ResizeAlgorithm::RESIZE_CUBIC)
                .scale(255.0f)
                .mean({0.485f, 0.456f, 0.406f})
                .scale({0.229f, 0.224f, 0.225f});
            ppp.input(input.get_any_name()).model().set_layout("NCHW");
            model = ppp.build();
            context_ = std::make_unique<ov::intel_gpu::ocl::D3DContext>(
                *core_, device);
            (void)device_name;
            compiled_ = std::make_unique<ov::CompiledModel>(
                core_->compile_model(model, *context_));
            request_ = std::make_unique<ov::InferRequest>(
                compiled_->create_infer_request());
            return true;
        } catch (const std::exception& exception) {
            error = std::string("depth_openvino_init:") + exception.what();
            reset();
            return false;
        }
    }

    bool infer(ID3D11Texture2D* nv12_surface, ID3D11Buffer* output,
               std::string& error) {
        if (!request_ || !context_ || !nv12_surface || !output) {
            error = "depth_openvino_not_ready"; return false;
        }
        try {
            // These are remote views, not pixel copies or allocations.
            auto planes = context_->create_tensor_nv12(source_height_,
                                                        source_width_,
                                                        nv12_surface);
            request_->set_input_tensor(0, planes.first);
            request_->set_input_tensor(1, planes.second);
            auto remote_output = context_->create_tensor(
                ov::element::f32, {1, 518, 518}, output);
            request_->set_output_tensor(0, remote_output);
            request_->infer();
            return true;
        } catch (const std::exception& exception) {
            error = std::string("depth_openvino_infer:") + exception.what();
            return false;
        }
    }

    void reset() {
        request_.reset();
        compiled_.reset();
        rgb_inputs_.clear();rgb_outputs_.clear();rgb_context_.reset();rgb_queue_.reset();
        context_.reset();
        core_.reset();
    }

    uint64_t rgb_acquire_count=0,rgb_release_wait_count=0;
    double rgb_release_wait_seconds=0;
private:
    struct RgbInput {ov::RemoteTensor shared,stable;};
    std::map<ID3D11Buffer*,RgbInput> rgb_inputs_;
    std::map<ID3D11Buffer*,RgbInput> rgb_outputs_;
    std::unique_ptr<cl::CommandQueue> rgb_queue_;
    std::unique_ptr<ov::intel_gpu::ocl::ClContext> rgb_context_;
    clEnqueueAcquireD3D11ObjectsKHR_fn acquire_=nullptr;
    clEnqueueReleaseD3D11ObjectsKHR_fn release_=nullptr;
    std::unique_ptr<ov::Core> core_;
    std::unique_ptr<ov::CompiledModel> compiled_;
    std::unique_ptr<ov::InferRequest> request_;
    std::unique_ptr<ov::intel_gpu::ocl::D3DContext> context_;
    size_t source_width_ = 0;
    size_t source_height_ = 0;
};

} // namespace xess_full_gpu_quality

#else

#define XESS_FG_QUALITY_HAS_OPENVINO 0
#include <d3d11.h>
#include <string>

namespace xess_full_gpu_quality {

// Capability-failure ABI.  The quality executable fails closed when this is
// selected and can never emit a pass with constant depth.
class CachedDepthAnything {
public:
    bool initialize(const std::string&, ID3D11Device*, const std::string&,
                    size_t, size_t, std::string& error) {
        error = "openvino_headers_unavailable"; return false;
    }
    bool infer(ID3D11Texture2D*, ID3D11Buffer*, std::string& error) {
        error = "openvino_headers_unavailable"; return false;
    }
    bool initialize_preprocessed(const std::string&,ID3D11Device*,std::string& error) {
        error="openvino_headers_unavailable";return false;
    }
    bool infer_preprocessed(ID3D11Buffer*,ID3D11Buffer*,std::string& error) {
        error="openvino_headers_unavailable";return false;
    }
    uint64_t rgb_acquire_count=0,rgb_release_wait_count=0;
    double rgb_release_wait_seconds=0;
};

} // namespace xess_full_gpu_quality

#endif
