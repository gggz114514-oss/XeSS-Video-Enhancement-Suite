#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "gpu_frame_contract.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cmath>
#include <fstream>
#include <vector>

// GPU-only conversion and RGB-u8 cubic/normalization. Coefficients are geometry
// metadata, generated once; never derived from CPU pixels. All per-slot resources
// persist. A D3D11 GPU copy exports the tensor to a bind=0 DX_BUFFER allocation
// supported by OpenVINO. The inference acquire/release owns that buffer until
// synchronous infer completes; D3D12 consumers own the separately shared output.
struct NativeDepthRgbPreprocess {
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    struct Slot {
        Ptr<ID3D11Texture2D> nv12,rgb,resized;
        Ptr<ID3D11ShaderResourceView> y,uv,rgb_srv;
        Ptr<ID3D11UnorderedAccessView> rgb_uav,resized_uav,tensor_uav;
        Ptr<ID3D11Buffer> tensor,remote;
    };
    Ptr<ID3D11Device> device;
    Ptr<ID3D11DeviceContext> context;
    Ptr<ID3D11ComputeShader> color_shader,tensor_shader;
    Ptr<ID3D11Buffer> weights,offsets,normalization,params,crop_params;
    Ptr<ID3D11ShaderResourceView> weights_srv,offsets_srv,normalization_srv;
    std::vector<Slot> slots;
    UINT width=0,height=0,allocation_w=0,allocation_h=0;
    uint64_t nv12_copy_count=0,tensor_copy_count=0,dispatch_count=0;
    static constexpr UINT model_size=518;
    bool check(HRESULT hr,const char* label,std::string& error) {
        if(SUCCEEDED(hr))return true;error=std::string(label)+":"+std::to_string(static_cast<unsigned>(hr));return false;
    }
    bool shader(const std::string& directory,const char* name,Ptr<ID3D11ComputeShader>& out,std::string& error) {
        std::ifstream file(directory+"/"+name+".cso",std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});
        if(bytes.empty()){error=std::string("depth_preprocess_shader_missing:")+name;return false;}
        return check(device->CreateComputeShader(bytes.data(),bytes.size(),nullptr,&out),name,error);
    }
    bool init(ID3D11Device* d,UINT w,UINT h,UINT aw,UINT ah,UINT count,const std::string& shaders,std::string& error) {
        device=d;d->GetImmediateContext(&context);width=w;height=h;allocation_w=aw;allocation_h=ah;
        if(!w||!h||w>aw||h>ah||(aw&1)||(ah&1)||!count){error="depth_preprocess_geometry";return false;}
        if(!shader(shaders,"native_depth_rgb",color_shader,error)||!shader(shaders,"native_depth_rgb_tensor",tensor_shader,error))return false;
        std::vector<float> alpha(model_size*2*4);std::vector<int> positions(model_size*2);
        for(UINT axis=0;axis<2;++axis)for(UINT i=0;i<model_size;++i) {
            float x=static_cast<float>((i+.5)*(static_cast<double>(axis?h:w)/model_size)-.5);
            int s=static_cast<int>(std::floor(x));x-=s;positions[axis*model_size+i]=s;
            float* a=&alpha[(axis*model_size+i)*4];constexpr float A=-.75f;
            a[0]=((A*(x+1)-5*A)*(x+1)+8*A)*(x+1)-4*A;
            a[1]=((A+2)*x-(A+3))*x*x+1;
            a[2]=((A+2)*(1-x)-(A+3))*(1-x)*(1-x)+1;
            a[3]=1-a[0]-a[1]-a[2];
        }
        auto table=[&](const void* data,UINT bytes,UINT stride,Ptr<ID3D11Buffer>& buffer,Ptr<ID3D11ShaderResourceView>& view) {
            D3D11_BUFFER_DESC bd{};bd.ByteWidth=bytes;bd.Usage=D3D11_USAGE_IMMUTABLE;bd.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=stride;D3D11_SUBRESOURCE_DATA sd{data,0,0};
            return check(d->CreateBuffer(&bd,&sd,&buffer),"depth_coeff_buffer",error)&&check(d->CreateShaderResourceView(buffer.Get(),nullptr,&view),"depth_coeff_srv",error);
        };
        float norm[768],mean[3]={.485f,.456f,.406f},stddev[3]={.229f,.224f,.225f};
        // A 3x256 algorithm-constant table, not an uploaded image. Volatile
        // operands retain real scalar FP32 divides, matching NumPy u8->f32.
        volatile float divisor=255.0f;
        for(UINT channel=0;channel<3;++channel)for(UINT value=0;value<256;++value) {
            volatile float scaled=static_cast<float>(value)/divisor;
            volatile float shifted=scaled-mean[channel],denominator=stddev[channel];
            norm[channel*256+value]=shifted/denominator;
        }
        if(!table(alpha.data(),static_cast<UINT>(alpha.size()*4),16,weights,weights_srv)||
           !table(positions.data(),static_cast<UINT>(positions.size()*4),4,offsets,offsets_srv)||
           !table(norm,sizeof(norm),4,normalization,normalization_srv))return false;
        D3D11_BUFFER_DESC cb{};cb.ByteWidth=16;cb.Usage=D3D11_USAGE_DEFAULT;cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(!check(d->CreateBuffer(&cb,nullptr,&params),"depth_params",error)||!check(d->CreateBuffer(&cb,nullptr,&crop_params),"depth_crop_params",error))return false;
        slots.resize(count);
        for(auto& s:slots) {
            D3D11_TEXTURE2D_DESC td{};td.Width=aw;td.Height=ah;td.MipLevels=td.ArraySize=1;td.Format=DXGI_FORMAT_NV12;td.SampleDesc.Count=1;
            td.Usage=D3D11_USAGE_DEFAULT;td.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            if(!check(d->CreateTexture2D(&td,nullptr,&s.nv12),"depth_nv12_copy_texture",error))return false;
            D3D11_SHADER_RESOURCE_VIEW_DESC sd{};sd.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D;sd.Texture2D.MipLevels=1;sd.Format=DXGI_FORMAT_R8_UNORM;
            if(!check(d->CreateShaderResourceView(s.nv12.Get(),&sd,&s.y),"depth_y_srv",error))return false;
            sd.Format=DXGI_FORMAT_R8G8_UNORM;
            if(!check(d->CreateShaderResourceView(s.nv12.Get(),&sd,&s.uv),"depth_uv_srv",error))return false;
            td.Width=w;td.Height=h;td.Format=DXGI_FORMAT_R8G8B8A8_UNORM;td.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
            if(!check(d->CreateTexture2D(&td,nullptr,&s.rgb),"depth_rgb_texture",error)||
               !check(d->CreateShaderResourceView(s.rgb.Get(),nullptr,&s.rgb_srv),"depth_rgb_srv",error)||
               !check(d->CreateUnorderedAccessView(s.rgb.Get(),nullptr,&s.rgb_uav),"depth_rgb_uav",error))return false;
            td.Width=td.Height=model_size;
            if(!check(d->CreateTexture2D(&td,nullptr,&s.resized),"depth_resized_texture",error)||
               !check(d->CreateUnorderedAccessView(s.resized.Get(),nullptr,&s.resized_uav),"depth_resized_uav",error))return false;
            D3D11_BUFFER_DESC bd{};bd.ByteWidth=3*model_size*model_size*4;bd.Usage=D3D11_USAGE_DEFAULT;
            bd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
            if(!check(d->CreateBuffer(&bd,nullptr,&s.tensor),"depth_tensor_buffer",error))return false;
            D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_R32_TYPELESS;ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;
            ud.Buffer.NumElements=bd.ByteWidth/4;ud.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
            if(!check(d->CreateUnorderedAccessView(s.tensor.Get(),&ud,&s.tensor_uav),"depth_tensor_uav",error))return false;
            bd.BindFlags=0;bd.MiscFlags=D3D11_RESOURCE_MISC_SHARED;
            if(!check(d->CreateBuffer(&bd,nullptr,&s.remote),"depth_remote_input_buffer",error))return false;
        }
        return true;
    }
    bool run(UINT index,ID3D11Texture2D* input,const xess_gpu::FrameMetadata& metadata,const xess_gpu::Crop& crop,std::string& error) {
        if(index>=slots.size()||!input){error="depth_preprocess_slot";return false;}
        if(metadata.range!=xess_gpu::Range::Limited || (metadata.matrix!=xess_gpu::Matrix::Bt601&&metadata.matrix!=xess_gpu::Matrix::Bt709)||
           metadata.transfer==xess_gpu::Transfer::Pq||metadata.transfer==xess_gpu::Transfer::Hlg) {error="depth_preprocess_unsupported_color_profile";return false;}
        D3D11_TEXTURE2D_DESC td{};input->GetDesc(&td);
        if(td.Format!=DXGI_FORMAT_NV12||td.ArraySize!=1||td.Width!=allocation_w||td.Height!=allocation_h||
           crop.width!=width||crop.height!=height||crop.x>allocation_w-width||crop.y>allocation_h-height) {error="depth_preprocess_input_geometry";return false;}
        auto& s=slots[index];context->CopyResource(s.nv12.Get(),input);++nv12_copy_count;
        UINT c[4]={width,height,metadata.matrix==xess_gpu::Matrix::Bt709?1u:0u,0};context->UpdateSubresource(params.Get(),0,nullptr,c,0,0);
        UINT offset[4]={crop.x,crop.y,0,0};context->UpdateSubresource(crop_params.Get(),0,nullptr,offset,0,0);
        ID3D11Buffer* cb[]={params.Get(),crop_params.Get()};context->CSSetConstantBuffers(0,2,cb);
        ID3D11ShaderResourceView* srv[]={s.y.Get(),s.uv.Get(),nullptr,nullptr};context->CSSetShaderResources(0,4,srv);
        ID3D11UnorderedAccessView* uav[]={s.rgb_uav.Get(),nullptr};context->CSSetUnorderedAccessViews(0,2,uav,nullptr);
        context->CSSetShader(color_shader.Get(),nullptr,0);context->Dispatch((width+7)/8,(height+7)/8,1);++dispatch_count;
        ID3D11UnorderedAccessView* nulluav[]={nullptr,nullptr};context->CSSetUnorderedAccessViews(0,2,nulluav,nullptr);
        srv[0]=s.rgb_srv.Get();srv[1]=weights_srv.Get();srv[2]=offsets_srv.Get();srv[3]=normalization_srv.Get();context->CSSetShaderResources(0,4,srv);
        c[2]=model_size;context->UpdateSubresource(params.Get(),0,nullptr,c,0,0);
        uav[0]=s.tensor_uav.Get();uav[1]=s.resized_uav.Get();context->CSSetUnorderedAccessViews(0,2,uav,nullptr);
        context->CSSetShader(tensor_shader.Get(),nullptr,0);context->Dispatch((model_size+7)/8,(model_size+7)/8,1);++dispatch_count;
        context->CSSetUnorderedAccessViews(0,2,nulluav,nullptr);
        ID3D11ShaderResourceView* nullsrv[]={nullptr,nullptr,nullptr,nullptr};context->CSSetShaderResources(0,4,nullsrv);context->CSSetShader(nullptr,nullptr,0);
        context->CopyResource(s.remote.Get(),s.tensor.Get());++tensor_copy_count;
        context->Flush(); // Submit D3D11 producer before OpenCL acquires DX_BUFFER.
        return true;
    }
};
