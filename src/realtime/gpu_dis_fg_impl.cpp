// P5 probe: oneVPL D3D11-owned NV12 decode surface -> same-LUID D3D12 shared
// handle -> complete OpenCV-DIS GPU graph (P2/P4 verified semantics: R8->R32F
// base, INTER_AREA pyramid, central-difference gradients, structure tensors,
// 8x8 patch inverse search with ordered waves, densification, full
// VariationalRefinement with 50 red/black SOR parity passes per level) ->
// OpenCV-contract final bridge -> XeSS (high-res MV). Optional QSV encode
// bridge (ported from the committed 3bcbc22 chain contract) and optional
// acceptance-only diagnostics (decoded luma dump, DIS flow readback, XeSS
// output readback, CPU-DIS-flow upload for the reference endpoint).
//
// The product path has zero full-frame CPU upload/readback; every diagnostic
// byte is counted separately and reported as acceptance traffic.
#ifndef XESS_FG_DIS_EMBED
#define main gate1_surface_probe_main
#include "vpl_decode_surface_probe.cpp"
#undef main
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <psapi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xess/xess.h"
#include "xess/xess_d3d12.h"

using Microsoft::WRL::ComPtr;

namespace {

// ---------------------------------------------------------------------------
// Small D3D12 helpers (same contract as the committed P2 diagnostic probe).
struct R {
    ComPtr<ID3D12Resource> p;
    D3D12_RESOURCE_STATES s = D3D12_RESOURCE_STATE_COMMON;
};
struct RB {
    R r;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total_bytes = 0;
};

std::string p5_hr(HRESULT h) {
    char b[32];
    std::snprintf(b, sizeof(b), "0x%08lX", static_cast<unsigned long>(h));
    return b;
}
bool p5_ok(HRESULT h, const char *what, std::string &err) {
    if (SUCCEEDED(h)) return true;
    err = std::string(what) + "=" + p5_hr(h);
    return false;
}
D3D12_HEAP_PROPERTIES p5_heap(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES p{};
    p.Type = type;
    p.CreationNodeMask = 1;
    p.VisibleNodeMask = 1;
    return p;
}
bool make_tex(ID3D12Device *d, UINT w, UINT h, DXGI_FORMAT f,
              D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags, R &out,
              std::string &err) {
    D3D12_RESOURCE_DESC dsc{};
    dsc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    dsc.Width = w;
    dsc.Height = h;
    dsc.DepthOrArraySize = 1;
    dsc.MipLevels = 1;
    dsc.Format = f;
    dsc.SampleDesc = {1, 0};
    dsc.Flags = flags;
    out.s = state;
    const D3D12_HEAP_PROPERTIES props = p5_heap(D3D12_HEAP_TYPE_DEFAULT);
    return p5_ok(d->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &dsc, state,
                                            nullptr, IID_PPV_ARGS(&out.p)),
                 "CreateCommittedResource(texture)", err);
}
bool make_buf(ID3D12Device *d, UINT64 bytes, D3D12_HEAP_TYPE type,
              D3D12_RESOURCE_STATES state, R &out, std::string &err) {
    D3D12_RESOURCE_DESC dsc{};
    dsc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    dsc.Width = bytes;
    dsc.Height = 1;
    dsc.DepthOrArraySize = 1;
    dsc.MipLevels = 1;
    dsc.SampleDesc = {1, 0};
    dsc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    out.s = state;
    const D3D12_HEAP_PROPERTIES props = p5_heap(type);
    return p5_ok(d->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &dsc, state,
                                            nullptr, IID_PPV_ARGS(&out.p)),
                 "CreateCommittedResource(buffer)", err);
}
void trans(ID3D12GraphicsCommandList *cl, R &r, D3D12_RESOURCE_STATES after) {
    if (r.s == after) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r.p.Get();
    b.Transition.StateBefore = r.s;
    b.Transition.StateAfter = after;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cl->ResourceBarrier(1, &b);
    r.s = after;
}
void uav_barrier(ID3D12GraphicsCommandList *cl, ID3D12Resource *r) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = r;
    cl->ResourceBarrier(1, &b);
}
D3D12_CPU_DESCRIPTOR_HANDLE cpu_h(ID3D12DescriptorHeap *h, UINT i, UINT inc) {
    auto x = h->GetCPUDescriptorHandleForHeapStart();
    x.ptr += static_cast<SIZE_T>(i) * inc;
    return x;
}
D3D12_GPU_DESCRIPTOR_HANDLE gpu_h(ID3D12DescriptorHeap *h, UINT i, UINT inc) {
    auto x = h->GetGPUDescriptorHandleForHeapStart();
    x.ptr += static_cast<UINT64>(i) * inc;
    return x;
}
void mk_srv(ID3D12Device *d, ID3D12DescriptorHeap *h, UINT i, UINT inc,
            ID3D12Resource *r, DXGI_FORMAT format, UINT plane = 0) {
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipLevels = 1;
    desc.Texture2D.PlaneSlice = plane;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d->CreateShaderResourceView(r, &desc, cpu_h(h, i, inc));
}
void mk_uav(ID3D12Device *d, ID3D12DescriptorHeap *h, UINT i, UINT inc,
            ID3D12Resource *r, DXGI_FORMAT format) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    d->CreateUnorderedAccessView(r, nullptr, &desc, cpu_h(h, i, inc));
}
bool read_blob(const std::string &path, std::vector<uint8_t> &blob, std::string &err) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        err = "shader_open=" + path;
        return false;
    }
    const std::streamoff size = in.tellg();
    if (size <= 0) {
        err = "shader_empty=" + path;
        return false;
    }
    blob.resize(static_cast<size_t>(size));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char *>(blob.data()), size);
    if (!in) {
        err = "shader_read=" + path;
        return false;
    }
    return true;
}
bool make_root(ID3D12Device *dev, UINT srv_count, UINT uav_count, UINT constants,
               ComPtr<ID3D12RootSignature> &out, std::string &err) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = srv_count;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = uav_count;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.Num32BitValues = constants;
    params[2].Constants.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 3;
    desc.pParameters = params;
    ComPtr<ID3DBlob> serialized, errors;
    if (!p5_ok(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &serialized, &errors),
               "serialize root signature", err))
        return false;
    return p5_ok(dev->CreateRootSignature(0, serialized->GetBufferPointer(),
                                          serialized->GetBufferSize(),
                                          IID_PPV_ARGS(&out)),
                 "create root signature", err);
}
bool make_pso(ID3D12Device *dev, ID3D12RootSignature *root,
              const std::vector<uint8_t> &shader, ComPtr<ID3D12PipelineState> &out,
              std::string &err) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.CS.pShaderBytecode = shader.data();
    desc.CS.BytecodeLength = shader.size();
    return p5_ok(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&out)),
                 "create compute pso", err);
}
bool make_readback_buf(ID3D12Device *d, ID3D12Resource *src, RB &out, std::string &err) {
    const auto desc = src->GetDesc();
    UINT64 total = 0;
    d->GetCopyableFootprints(&desc, 0, 1, 0, &out.fp, nullptr, nullptr, &total);
    out.total_bytes = total;
    return make_buf(d, total, D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_STATE_COPY_DEST, out.r, err);
}
bool write_packed(ID3D12Resource *readback, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT &fp,
                  UINT width, UINT height, UINT bytes_per_pixel, const std::string &path,
                  std::string &err) {
    uint8_t *mapped = nullptr;
    const D3D12_RANGE no_read{0, 0};
    if (!p5_ok(readback->Map(0, &no_read, reinterpret_cast<void **>(&mapped)),
               "map readback", err))
        return false;
    std::vector<uint8_t> packed(static_cast<size_t>(width) * height * bytes_per_pixel);
    for (UINT y = 0; y < height; ++y)
        std::memcpy(packed.data() + static_cast<size_t>(y) * width * bytes_per_pixel,
                    mapped + fp.Offset + static_cast<size_t>(y) * fp.Footprint.RowPitch,
                    static_cast<size_t>(width) * bytes_per_pixel);
    readback->Unmap(0, nullptr);
    FILE *f = nullptr;
    if (fopen_s(&f, path.c_str(), "wb") || !f) {
        err = "open write " + path;
        return false;
    }
    const bool good = fwrite(packed.data(), 1, packed.size(), f) == packed.size();
    fclose(f);
    if (!good) err = "write " + path;
    return good;
}
bool wait_fence(ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value,
                std::string &err) {
    if (fence->GetCompletedValue() >= value) return true;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) {
        err = "CreateEvent";
        return false;
    }
    const HRESULT h = fence->SetEventOnCompletion(value, event);
    if (SUCCEEDED(h)) WaitForSingleObject(event, 60000);
    CloseHandle(event);
    if (FAILED(h) || fence->GetCompletedValue() < value) {
        err = "fence timeout";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Complete GPU DIS graph. Two role-swapped pyramid chains (A/B): the current
// frame's chain is recomputed from the freshly converted luma; the previous
// frame's chain is kept from the prior frame (identical values to a cold
// INTER_AREA recomputation of the same level-0 texture). Per-pair scratch
// (gradient/sparse/dense/initial/VR) is a single persistent set.
struct DisLevel {
    UINT width = 0, height = 0, sparse_width = 0, sparse_height = 0;
    R cur_a, cur_b;  // level texture, chain A and chain B
    R gradient, sparse, dense, initial;
    std::array<R, 5> tensors;
    R vr_work, vr_dflow, vr_dflow_alt;
    std::array<R, 8> vr_deriv;
    std::array<R, 6> vr_coeff;
};

struct DisPipeline {
    ID3D12Device *device = nullptr;
    UINT crop_w = 0, crop_h = 0, out_w = 0, out_h = 0;
    UINT finest_scale = 2, coarsest_scale = 2;
    std::vector<DisLevel> levels;  // indexed 0..coarsest_scale

    ComPtr<ID3D12RootSignature> base_root, down_root, gradient_root, structure_root,
        patch_root, densify_root, upsample_root, vr_deriv1_root, vr_deriv2_root,
        vr_system_root, vr_sor_root, vr_update_root, vr_zero_root, bridge_root,
        color_root, mask_root;
    ComPtr<ID3D12PipelineState> base_pso, down_pso, gradient_pso, structure_pso,
        patch_pso, densify_pso, upsample_pso, vr_deriv1_pso, vr_deriv2_pso,
        vr_system_pso, vr_sor_pso, vr_update_pso, vr_zero_pso, bridge_pso,
        color_pso, mask_pso;

    // heaps: role-dependent stages come in A/B variants; role-independent
    // stages are single. (role 0: current=chain A, previous=chain B)
    ComPtr<ID3D12DescriptorHeap> base_heap[2], misc_heap;
    std::vector<ComPtr<ID3D12DescriptorHeap>> down_heap[2], gradient_heap[2],
        patch_heap[2], densify_heap[2], vr_deriv1_heap[2], structure_heap,
        upsample_heap, vr_deriv2_heap, vr_system_heap, vr_sor_heap, vr_sor_alt_heap,
        vr_update_heap, vr_zero_heap;
    UINT inc = 0;

    // XeSS-facing per-frame resources (single, strictly serial frames)
    R color, velocity, mask, diag_flow, cpu_flow;
    R luma_a, luma_b;
    RB diag_flow_rb;
    bool dump_levels = false;
    std::vector<RB> vr_rb, dense_rb, sparse_rb, chain_a_rb, chain_b_rb, grad_rb, tensor_rb;
    R cpu_flow_upload;

    static constexpr UINT desc_color = 0, desc_luma_uav = 1, desc_prev_luma_srv = 2,
                          desc_cur_luma_srv = 3, desc_velocity_uav = 4,
                          desc_mask_uav = 5, desc_nv12_luma = 6, desc_nv12_chroma = 7,
                          desc_level_flow_srv = 8, desc_cpu_flow_srv = 9,
                          desc_diag_flow_uav = 10;
    UINT base_slot_srv[2] = {0, 0}, base_slot_uav[2] = {1, 1};

    UINT64 product_gpu_copies = 0;   // explicit GPU copies on the product path
    UINT64 acceptance_readback_bytes = 0;
    UINT64 acceptance_upload_bytes = 0;
    UINT parity_dispatches = 0, patch_dispatches = 0;

    std::string err;

    bool init(ID3D12Device *dev, const std::string &shader_dir, UINT cw, UINT ch,
              UINT ow, UINT oh) {
        device = dev;
        crop_w = cw;
        crop_h = ch;
        out_w = ow;
        out_h = oh;
        inc = device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        const double lw = static_cast<double>(std::max(crop_w, crop_h));
        const double sw = static_cast<double>(std::min(crop_w, crop_h));
        const int max_scale = static_cast<int>(std::min(
            std::floor(std::log(lw / 32.0) / std::log(2.0) + 0.5),
            std::floor(std::log(sw / 8.0) / std::log(2.0))));
        if (max_scale < static_cast<int>(finest_scale)) {
            err = "input_too_small_for_dis_finest_scale_2";
            return false;
        }
        coarsest_scale = static_cast<UINT>(max_scale);

        struct Stage {
            const char *name;
            UINT srv, uav, cbuf;
            ComPtr<ID3D12RootSignature> *root;
            ComPtr<ID3D12PipelineState> *pso;
        };
        const std::vector<Stage> stages = {
            {"dis_d2_base", 1, 1, 4, &base_root, &base_pso},
            {"dis_perfect_area", 1, 1, 4, &down_root, &down_pso},
            {"dis_d2_gradient", 1, 1, 4, &gradient_root, &gradient_pso},
            {"dis_d3_structure", 1, 5, 4, &structure_root, &structure_pso},
            {"dis_perfect_patch", 9, 1, 10, &patch_root, &patch_pso},
            {"dis_perfect_densify", 3, 1, 6, &densify_root, &densify_pso},
            {"dis_perfect_upsample", 1, 1, 4, &upsample_root, &upsample_pso},
            {"dis_perfect_vr_deriv1", 3, 5, 2, &vr_deriv1_root, &vr_deriv1_pso},
            {"dis_perfect_vr_deriv2", 2, 3, 2, &vr_deriv2_root, &vr_deriv2_pso},
            {"dis_perfect_vr_system", 11, 6, 6, &vr_system_root, &vr_system_pso},
            {"dis_perfect_vr_sor", 7, 1, 4, &vr_sor_root, &vr_sor_pso},
            {"dis_perfect_vr_update", 2, 1, 2, &vr_update_root, &vr_update_pso},
            {"dis_perfect_vr_zero", 1, 1, 2, &vr_zero_root, &vr_zero_pso},
            {"dis_final_bridge", 2, 1, 12, &bridge_root, &bridge_pso},
            {"surface_nv12_color", 2, 2, 4, &color_root, &color_pso},
            {"surface_gpu_mask", 2, 2, 4, &mask_root, &mask_pso},
        };
        for (const Stage &st : stages) {
            std::vector<uint8_t> blob;
            if (!read_blob(shader_dir + "/" + st.name + ".dxil", blob, err)) return false;
            if (!make_root(device, st.srv, st.uav, st.cbuf, *st.root, err)) return false;
            if (!make_pso(device, st.root->Get(), blob, *st.pso, err)) return false;
        }

        const UINT level_count = coarsest_scale + 1;
        levels.resize(level_count);
        levels[0].width = crop_w;
        levels[0].height = crop_h;
        for (UINT i = 1; i <= coarsest_scale; ++i) {
            levels[i].width = std::max<UINT>(1, levels[i - 1].width / 2);
            levels[i].height = std::max<UINT>(1, levels[i - 1].height / 2);
        }
        for (UINT i = 0; i <= coarsest_scale; ++i) {
            DisLevel &l = levels[i];
            l.sparse_width = 1 + (l.width - 8) / 4;
            l.sparse_height = 1 + (l.height - 8) / 4;
            auto rr = [&](UINT w, UINT h, DXGI_FORMAT f, R &out) {
                return make_tex(device, w, h, f, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, out, err);
            };
            if (!rr(l.width, l.height, DXGI_FORMAT_R32_FLOAT, l.cur_a) ||
                !rr(l.width, l.height, DXGI_FORMAT_R32_FLOAT, l.cur_b) ||
                !rr(l.width, l.height, DXGI_FORMAT_R32G32_SINT, l.gradient) ||
                !rr(l.sparse_width, l.sparse_height, DXGI_FORMAT_R32G32_FLOAT, l.sparse) ||
                !rr(l.width, l.height, DXGI_FORMAT_R32G32_FLOAT, l.dense) ||
                !rr(l.width, l.height, DXGI_FORMAT_R32G32_FLOAT, l.initial)) {
                return false;
            }
            for (auto &t : l.tensors)
                if (!rr(l.sparse_width, l.sparse_height, DXGI_FORMAT_R32_FLOAT, t))
                    return false;
            if (!rr(l.width, l.height, DXGI_FORMAT_R32G32_FLOAT, l.vr_work) ||
                !rr(l.width, l.height, DXGI_FORMAT_R32G32_FLOAT, l.vr_dflow) ||
                !rr(l.width, l.height, DXGI_FORMAT_R32G32_FLOAT, l.vr_dflow_alt))
                return false;
            for (auto &dv : l.vr_deriv)
                if (!rr(l.width, l.height, DXGI_FORMAT_R32_FLOAT, dv)) return false;
            for (auto &c : l.vr_coeff)
                if (!rr(l.width, l.height, DXGI_FORMAT_R32_FLOAT, c)) return false;
        }
        if (!make_tex(device, crop_w, crop_h, DXGI_FORMAT_R8G8B8A8_UNORM,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, color, err) ||
            !make_tex(device, crop_w, crop_h, DXGI_FORMAT_R32_FLOAT,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, luma_a, err) ||
            !make_tex(device, crop_w, crop_h, DXGI_FORMAT_R32_FLOAT,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, luma_b, err) ||
            !make_tex(device, out_w, out_h, DXGI_FORMAT_R16G16_FLOAT,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, velocity, err) ||
            !make_tex(device, crop_w, crop_h, DXGI_FORMAT_R8_UNORM,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, mask, err) ||
            !make_tex(device, crop_w, crop_h, DXGI_FORMAT_R32G32_FLOAT,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, diag_flow, err) ||
            !make_tex(device, crop_w, crop_h, DXGI_FORMAT_R32G32_FLOAT,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, cpu_flow, err)) {
            return false;
        }
        if (!make_readback_buf(device, diag_flow.p.Get(), diag_flow_rb, err)) return false;
        // acceptance: per-level VR/dense/sparse readbacks (diagnostic only)
        if (dump_levels) {
            vr_rb.resize(level_count);
            dense_rb.resize(level_count);
            sparse_rb.resize(level_count);
            chain_a_rb.resize(level_count);
            chain_b_rb.resize(level_count);
            grad_rb.resize(level_count);
            tensor_rb.resize(level_count);
            for (UINT i = 0; i <= coarsest_scale; ++i) {
                if (!make_readback_buf(device, levels[i].vr_work.p.Get(), vr_rb[i], err) ||
                    !make_readback_buf(device, levels[i].dense.p.Get(), dense_rb[i], err) ||
                    !make_readback_buf(device, levels[i].sparse.p.Get(), sparse_rb[i], err) ||
                    !make_readback_buf(device, levels[i].cur_a.p.Get(), chain_a_rb[i], err) ||
                    !make_readback_buf(device, levels[i].cur_b.p.Get(), chain_b_rb[i], err) ||
                    !make_readback_buf(device, levels[i].gradient.p.Get(), grad_rb[i], err) ||
                    !make_readback_buf(device, levels[i].tensors[0].p.Get(), tensor_rb[i], err))
                    return false;
            }
        }
        {
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
            UINT64 total = 0;
            device->GetCopyableFootprints(&cpu_flow.p->GetDesc(), 0, 1, 0, &fp, nullptr,
                                          nullptr, &total);
            if (!make_buf(device, total, D3D12_HEAP_TYPE_UPLOAD,
                          D3D12_RESOURCE_STATE_GENERIC_READ, cpu_flow_upload, err))
                return false;
        }

        auto new_heap = [&](UINT count, ComPtr<ID3D12DescriptorHeap> &heap) {
            D3D12_DESCRIPTOR_HEAP_DESC d{};
            d.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            d.NumDescriptors = count;
            d.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            return p5_ok(device->CreateDescriptorHeap(&d, IID_PPV_ARGS(&heap)),
                         "CreateDescriptorHeap", err);
        };
        for (UINT v = 0; v < 2; ++v) {
            down_heap[v].resize(level_count);
            gradient_heap[v].resize(level_count);
            patch_heap[v].resize(level_count);
            densify_heap[v].resize(level_count);
            vr_deriv1_heap[v].resize(level_count);
        }
        structure_heap.resize(level_count);
        upsample_heap.resize(level_count);
        vr_deriv2_heap.resize(level_count);
        vr_system_heap.resize(level_count);
        vr_sor_heap.resize(level_count);
        vr_sor_alt_heap.resize(level_count);
        vr_update_heap.resize(level_count);
        vr_zero_heap.resize(level_count);
        for (UINT i = finest_scale; i <= coarsest_scale; ++i) {
            if (!new_heap(6, structure_heap[i]) || !new_heap(2, upsample_heap[i]) ||
                !new_heap(5, vr_deriv2_heap[i]) || !new_heap(17, vr_system_heap[i]) ||
                !new_heap(8, vr_sor_heap[i]) || !new_heap(8, vr_sor_alt_heap[i]) ||
                !new_heap(3, vr_update_heap[i]) || !new_heap(2, vr_zero_heap[i]))
                return false;
        }
        for (UINT v = 0; v < 2; ++v) {
            if (!new_heap(4, base_heap[v])) return false;
            for (UINT i = finest_scale; i <= coarsest_scale; ++i) {
                if (!new_heap(4, down_heap[v][i]) || !new_heap(2, gradient_heap[v][i]) ||
                    !new_heap(10, patch_heap[v][i]) || !new_heap(4, densify_heap[v][i]) ||
                    !new_heap(8, vr_deriv1_heap[v][i]))
                    return false;
            }
        }
        if (!new_heap(16, misc_heap)) return false;
        bind_static_descriptors();
        return true;
    }

    // Role-independent descriptors.
    void bind_static_descriptors() {
        mk_uav(device, misc_heap.Get(), desc_color, inc, color.p.Get(),
               DXGI_FORMAT_R8G8B8A8_UNORM);
        mk_uav(device, misc_heap.Get(), desc_velocity_uav, inc, velocity.p.Get(),
               DXGI_FORMAT_R16G16_FLOAT);
        mk_uav(device, misc_heap.Get(), desc_mask_uav, inc, mask.p.Get(),
               DXGI_FORMAT_R8_UNORM);
        mk_uav(device, misc_heap.Get(), desc_diag_flow_uav, inc, diag_flow.p.Get(),
               DXGI_FORMAT_R32G32_FLOAT);
        mk_srv(device, misc_heap.Get(), desc_level_flow_srv, inc,
               levels[finest_scale].vr_work.p.Get(), DXGI_FORMAT_R32G32_FLOAT);
        mk_srv(device, misc_heap.Get(), desc_cpu_flow_srv, inc, cpu_flow.p.Get(),
               DXGI_FORMAT_R32G32_FLOAT);
        for (UINT i = finest_scale; i <= coarsest_scale; ++i) {
            DisLevel &l = levels[i];
            mk_srv(device, structure_heap[i].Get(), 0, inc, l.gradient.p.Get(),
                   DXGI_FORMAT_R32G32_SINT);
            for (UINT k = 0; k < 5; ++k)
                mk_uav(device, structure_heap[i].Get(), 1 + k, inc, l.tensors[k].p.Get(),
                       DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, upsample_heap[i].Get(), 0, inc, l.vr_work.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            if (i > finest_scale)
                mk_uav(device, upsample_heap[i].Get(), 1, inc, levels[i - 1].initial.p.Get(),
                       DXGI_FORMAT_R32G32_FLOAT);
            mk_srv(device, vr_deriv2_heap[i].Get(), 0, inc, l.vr_deriv[0].p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, vr_deriv2_heap[i].Get(), 1, inc, l.vr_deriv[1].p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            for (UINT k = 0; k < 3; ++k)
                mk_uav(device, vr_deriv2_heap[i].Get(), 2 + k, inc,
                       l.vr_deriv[5 + k].p.Get(), DXGI_FORMAT_R32_FLOAT);
            // vr_deriv layout: deriv1 writes [0]=Ix [1]=Iy [2]=Iz [3]=Ixz
            // [4]=Iyz; deriv2 writes [5]=Ixx [6]=Ixy [7]=Iyy. The system
            // shader expects t0..t7 = Ix,Iy,Iz,Ixx,Ixy,Iyy,Ixz,Iyz, so the
            // last five SRVs follow the {0,1,2,5,6,7,3,4} permutation (P4 fix).
            static const UINT kSystemDerivOrder[8] = {0, 1, 2, 5, 6, 7, 3, 4};
            for (UINT k = 0; k < 8; ++k)
                mk_srv(device, vr_system_heap[i].Get(), k, inc,
                       l.vr_deriv[kSystemDerivOrder[k]].p.Get(), DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, vr_system_heap[i].Get(), 8, inc, l.vr_dflow.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_srv(device, vr_system_heap[i].Get(), 9, inc, l.dense.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_srv(device, vr_system_heap[i].Get(), 10, inc, l.vr_work.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            for (UINT k = 0; k < 6; ++k)
                mk_uav(device, vr_system_heap[i].Get(), 11 + k, inc, l.vr_coeff[k].p.Get(),
                       DXGI_FORMAT_R32_FLOAT);
            for (UINT k = 0; k < 6; ++k)
                mk_srv(device, vr_sor_heap[i].Get(), k, inc, l.vr_coeff[k].p.Get(),
                       DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, vr_sor_heap[i].Get(), 6, inc, l.vr_dflow.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_uav(device, vr_sor_heap[i].Get(), 7, inc, l.vr_dflow_alt.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            for (UINT k = 0; k < 6; ++k)
                mk_srv(device, vr_sor_alt_heap[i].Get(), k, inc, l.vr_coeff[k].p.Get(),
                       DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, vr_sor_alt_heap[i].Get(), 6, inc, l.vr_dflow_alt.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_uav(device, vr_sor_alt_heap[i].Get(), 7, inc, l.vr_dflow.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_srv(device, vr_update_heap[i].Get(), 0, inc, l.dense.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_srv(device, vr_update_heap[i].Get(), 1, inc, l.vr_dflow.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_uav(device, vr_update_heap[i].Get(), 2, inc, l.vr_work.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_srv(device, vr_zero_heap[i].Get(), 0, inc, l.vr_dflow.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_uav(device, vr_zero_heap[i].Get(), 1, inc, l.vr_dflow.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
        }
    }

    // Role-dependent descriptors for variant v (0: cur=A/prev=B, 1: swapped).
    // frame0_self binds both roles to chain A (frame-0 self pair).
    void bind_role_descriptors(UINT v, bool frame0_self, ID3D12Resource *imported_nv12) {
        R &cur0 = frame0_self ? levels[0].cur_a : (v == 0 ? levels[0].cur_a : levels[0].cur_b);
        R &prev0 = frame0_self ? levels[0].cur_a : (v == 0 ? levels[0].cur_b : levels[0].cur_a);
        mk_srv(device, base_heap[v].Get(), base_slot_srv[v], inc, imported_nv12,
               DXGI_FORMAT_R8_UNORM, 0);
        mk_uav(device, base_heap[v].Get(), base_slot_uav[v], inc, cur0.p.Get(),
               DXGI_FORMAT_R32_FLOAT);
        auto cur_of = [&](UINT i) -> R & {
            return frame0_self ? levels[i].cur_a
                               : (v == 0 ? levels[i].cur_a : levels[i].cur_b);
        };
        auto prev_of = [&](UINT i) -> R & {
            return frame0_self ? levels[i].cur_a
                               : (v == 0 ? levels[i].cur_b : levels[i].cur_a);
        };
        for (UINT i = finest_scale; i <= coarsest_scale; ++i) {
            R &src = (i == finest_scale) ? cur0 : cur_of(i - 1);
            R &psrc = (i == finest_scale) ? prev0 : prev_of(i - 1);
            mk_srv(device, down_heap[v][i].Get(), 0, inc, src.p.Get(), DXGI_FORMAT_R32_FLOAT);
            mk_uav(device, down_heap[v][i].Get(), 1, inc, cur_of(i).p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, down_heap[v][i].Get(), 2, inc, psrc.p.Get(), DXGI_FORMAT_R32_FLOAT);
            mk_uav(device, down_heap[v][i].Get(), 3, inc, prev_of(i).p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, gradient_heap[v][i].Get(), 0, inc, cur_of(i).p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            mk_uav(device, gradient_heap[v][i].Get(), 1, inc, levels[i].gradient.p.Get(),
                   DXGI_FORMAT_R32G32_SINT);
            mk_srv(device, patch_heap[v][i].Get(), 0, inc, cur_of(i).p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, patch_heap[v][i].Get(), 1, inc, prev_of(i).p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, patch_heap[v][i].Get(), 2, inc, levels[i].gradient.p.Get(),
                   DXGI_FORMAT_R32G32_SINT);
            for (UINT k = 0; k < 5; ++k)
                mk_srv(device, patch_heap[v][i].Get(), 3 + k, inc,
                       levels[i].tensors[k].p.Get(), DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, patch_heap[v][i].Get(), 8, inc, levels[i].initial.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_uav(device, patch_heap[v][i].Get(), 9, inc, levels[i].sparse.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_srv(device, densify_heap[v][i].Get(), 0, inc, cur_of(i).p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, densify_heap[v][i].Get(), 1, inc, prev_of(i).p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, densify_heap[v][i].Get(), 2, inc, levels[i].sparse.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_uav(device, densify_heap[v][i].Get(), 3, inc, levels[i].dense.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            mk_srv(device, vr_deriv1_heap[v][i].Get(), 0, inc, cur_of(i).p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, vr_deriv1_heap[v][i].Get(), 1, inc, prev_of(i).p.Get(),
                   DXGI_FORMAT_R32_FLOAT);
            mk_srv(device, vr_deriv1_heap[v][i].Get(), 2, inc, levels[i].vr_work.p.Get(),
                   DXGI_FORMAT_R32G32_FLOAT);
            for (UINT k = 0; k < 5; ++k)
                mk_uav(device, vr_deriv1_heap[v][i].Get(), 3 + k, inc,
                       levels[i].vr_deriv[k].p.Get(), DXGI_FORMAT_R32_FLOAT);
        }
    }

    void set_heap(ID3D12GraphicsCommandList *cl, ID3D12DescriptorHeap *heap) {
        ID3D12DescriptorHeap *hs[] = {heap};
        cl->SetDescriptorHeaps(1, hs);
    }

    // Per-frame NV12 plane SRVs (colour conversion) and the colour/luma
    // conversion dispatch. Always executed (XeSS colour input), even when the
    // DIS graph is skipped on the CPU-flow reference endpoint.
    void record_color(ID3D12GraphicsCommandList *cl, ID3D12Resource *imported_nv12,
                      R &luma_cur, UINT matrix_select = 0, UINT range_select = 0) {
        mk_srv(device, misc_heap.Get(), desc_nv12_luma, inc, imported_nv12,
               DXGI_FORMAT_R8_UNORM, 0);
        mk_srv(device, misc_heap.Get(), desc_nv12_chroma, inc, imported_nv12,
               DXGI_FORMAT_R8G8_UNORM, 1);
        mk_uav(device, misc_heap.Get(), desc_luma_uav, inc, luma_cur.p.Get(),
               DXGI_FORMAT_R32_FLOAT);
        trans(cl, color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        trans(cl, luma_cur, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        set_heap(cl, misc_heap.Get());
        cl->SetComputeRootSignature(color_root.Get());
        cl->SetPipelineState(color_pso.Get());
        const UINT cconst[4] = {crop_w, crop_h, matrix_select, range_select};
        cl->SetComputeRootDescriptorTable(0, gpu_h(misc_heap.Get(), desc_nv12_luma, inc));
        cl->SetComputeRootDescriptorTable(1, gpu_h(misc_heap.Get(), desc_color, inc));
        cl->SetComputeRoot32BitConstants(2, 4, cconst, 0);
        cl->Dispatch((crop_w + 7) / 8, (crop_h + 7) / 8, 1);
        trans(cl, color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        trans(cl, luma_cur, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    // Complete DIS graph for one frame pair (P2/P4 scheduler, streaming
    // previous-chain reuse). imported_nv12 is in NON_PIXEL_SHADER_RESOURCE.
    void record_graph(ID3D12GraphicsCommandList *cl, UINT v, bool frame0_self) {
        auto cur_of = [&](UINT i) -> R & {
            return frame0_self ? levels[i].cur_a
                               : (v == 0 ? levels[i].cur_a : levels[i].cur_b);
        };
        auto prev_of = [&](UINT i) -> R & {
            return frame0_self ? levels[i].cur_a
                               : (v == 0 ? levels[i].cur_b : levels[i].cur_a);
        };

        // base conversion: imported NV12 luma -> level-0 R32F (0..255 CV_8U)
        set_heap(cl, base_heap[v].Get());
        cl->SetComputeRootSignature(base_root.Get());
        cl->SetPipelineState(base_pso.Get());
        const UINT bconst[4] = {crop_w, crop_h, crop_w, crop_h};
        cl->SetComputeRootDescriptorTable(0, gpu_h(base_heap[v].Get(), base_slot_srv[v], inc));
        cl->SetComputeRootDescriptorTable(1, gpu_h(base_heap[v].Get(), base_slot_uav[v], inc));
        cl->SetComputeRoot32BitConstants(2, 4, bconst, 0);
        trans(cl, cur_of(0), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cl->Dispatch((crop_w + 7) / 8, (crop_h + 7) / 8, 1);
        trans(cl, cur_of(0), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // current-chain INTER_AREA downsample (previous chain is persistent)
        for (UINT i = finest_scale; i <= coarsest_scale; ++i) {
            set_heap(cl, down_heap[v][i].Get());
            cl->SetComputeRootSignature(down_root.Get());
            cl->SetPipelineState(down_pso.Get());
            const UINT p[4] = {i == finest_scale ? levels[0].width : levels[i - 1].width,
                               i == finest_scale ? levels[0].height : levels[i - 1].height,
                               levels[i].width, levels[i].height};
            cl->SetComputeRootDescriptorTable(0, gpu_h(down_heap[v][i].Get(), 0, inc));
            cl->SetComputeRootDescriptorTable(1, gpu_h(down_heap[v][i].Get(), 1, inc));
            cl->SetComputeRoot32BitConstants(2, 4, p, 0);
            trans(cl, cur_of(i), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cl->Dispatch((levels[i].width + 7) / 8, (levels[i].height + 7) / 8, 1);
            trans(cl, cur_of(i), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }

        for (int level = static_cast<int>(coarsest_scale);
             level >= static_cast<int>(finest_scale); --level) {
            const UINT i = static_cast<UINT>(level);
            DisLevel &l = levels[i];
            set_heap(cl, gradient_heap[v][i].Get());
            cl->SetComputeRootSignature(gradient_root.Get());
            cl->SetPipelineState(gradient_pso.Get());
            const UINT gp[4] = {l.width, l.height, l.width, l.height};
            cl->SetComputeRootDescriptorTable(0, gpu_h(gradient_heap[v][i].Get(), 0, inc));
            cl->SetComputeRootDescriptorTable(1, gpu_h(gradient_heap[v][i].Get(), 1, inc));
            cl->SetComputeRoot32BitConstants(2, 4, gp, 0);
            trans(cl, l.gradient, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cl->Dispatch((l.width + 7) / 8, (l.height + 7) / 8, 1);
            trans(cl, l.gradient, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            set_heap(cl, structure_heap[i].Get());
            cl->SetComputeRootSignature(structure_root.Get());
            cl->SetPipelineState(structure_pso.Get());
            const UINT sp[4] = {l.width, l.height, l.sparse_width, l.sparse_height};
            cl->SetComputeRootDescriptorTable(0, gpu_h(structure_heap[i].Get(), 0, inc));
            cl->SetComputeRootDescriptorTable(1, gpu_h(structure_heap[i].Get(), 1, inc));
            cl->SetComputeRoot32BitConstants(2, 4, sp, 0);
            for (auto &t : l.tensors) trans(cl, t, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cl->Dispatch((l.sparse_width + 7) / 8, (l.sparse_height + 7) / 8, 1);
            for (auto &t : l.tensors)
                trans(cl, t, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            set_heap(cl, patch_heap[v][i].Get());
            cl->SetComputeRootSignature(patch_root.Get());
            cl->SetPipelineState(patch_pso.Get());
            cl->SetComputeRootDescriptorTable(0, gpu_h(patch_heap[v][i].Get(), 0, inc));
            cl->SetComputeRootDescriptorTable(1, gpu_h(patch_heap[v][i].Get(), 9, inc));
            trans(cl, l.sparse, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            const UINT diagonals = l.sparse_width + l.sparse_height - 1;
            for (UINT direction = 0; direction < 2; ++direction) {
                for (UINT diagonal = 0; diagonal < diagonals; ++diagonal) {
                    const UINT first =
                        diagonal + 1 > l.sparse_width ? diagonal + 1 - l.sparse_width : 0;
                    const UINT last = std::min(diagonal, l.sparse_height - 1);
                    const UINT count = last >= first ? last - first + 1 : 0;
                    if (!count) continue;
                    const UINT pp[10] = {l.width, l.height, l.sparse_width, l.sparse_height,
                                         8, 4, 8, direction, diagonal,
                                         i < coarsest_scale ? 1u : 0u};
                    cl->SetComputeRoot32BitConstants(2, 10, pp, 0);
                    cl->Dispatch(count, 1, 1);
                    uav_barrier(cl, l.sparse.p.Get());
                    ++patch_dispatches;
                }
            }
            set_heap(cl, densify_heap[v][i].Get());
            cl->SetComputeRootSignature(densify_root.Get());
            cl->SetPipelineState(densify_pso.Get());
            const UINT dp[6] = {l.width, l.height, l.sparse_width, l.sparse_height, 8, 4};
            cl->SetComputeRootDescriptorTable(0, gpu_h(densify_heap[v][i].Get(), 0, inc));
            cl->SetComputeRootDescriptorTable(1, gpu_h(densify_heap[v][i].Get(), 3, inc));
            cl->SetComputeRoot32BitConstants(2, 6, dp, 0);
            trans(cl, l.dense, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cl->Dispatch((l.width + 7) / 8, (l.height + 7) / 8, 1);
            trans(cl, l.dense, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            // Complete VariationalRefinement: zero dflow, initial update,
            // derivatives, then 5 fixed-point iterations of (system + 10
            // red/black SOR parity passes + update). Ping-pong scheduling as
            // verified by P4; bit-identical to in-place when both land.
            set_heap(cl, vr_zero_heap[i].Get());
            cl->SetComputeRootSignature(vr_zero_root.Get());
            cl->SetPipelineState(vr_zero_pso.Get());
            const UINT wh[2] = {l.width, l.height};
            cl->SetComputeRootDescriptorTable(0, gpu_h(vr_zero_heap[i].Get(), 0, inc));
            cl->SetComputeRootDescriptorTable(1, gpu_h(vr_zero_heap[i].Get(), 1, inc));
            cl->SetComputeRoot32BitConstants(2, 2, wh, 0);
            trans(cl, l.vr_dflow, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cl->Dispatch((l.width + 7) / 8, (l.height + 7) / 8, 1);
            uav_barrier(cl, l.vr_dflow.p.Get());
            trans(cl, l.vr_dflow, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            set_heap(cl, vr_update_heap[i].Get());
            cl->SetComputeRootSignature(vr_update_root.Get());
            cl->SetPipelineState(vr_update_pso.Get());
            cl->SetComputeRootDescriptorTable(0, gpu_h(vr_update_heap[i].Get(), 0, inc));
            cl->SetComputeRootDescriptorTable(1, gpu_h(vr_update_heap[i].Get(), 2, inc));
            cl->SetComputeRoot32BitConstants(2, 2, wh, 0);
            trans(cl, l.vr_work, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cl->Dispatch((l.width + 7) / 8, (l.height + 7) / 8, 1);
            trans(cl, l.vr_work, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            set_heap(cl, vr_deriv1_heap[v][i].Get());
            cl->SetComputeRootSignature(vr_deriv1_root.Get());
            cl->SetPipelineState(vr_deriv1_pso.Get());
            cl->SetComputeRootDescriptorTable(0, gpu_h(vr_deriv1_heap[v][i].Get(), 0, inc));
            cl->SetComputeRootDescriptorTable(1, gpu_h(vr_deriv1_heap[v][i].Get(), 3, inc));
            cl->SetComputeRoot32BitConstants(2, 2, wh, 0);
            for (UINT k = 0; k < 5; ++k)
                trans(cl, l.vr_deriv[k], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cl->Dispatch((l.width + 7) / 8, (l.height + 7) / 8, 1);
            for (UINT k = 0; k < 5; ++k)
                trans(cl, l.vr_deriv[k], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            set_heap(cl, vr_deriv2_heap[i].Get());
            cl->SetComputeRootSignature(vr_deriv2_root.Get());
            cl->SetPipelineState(vr_deriv2_pso.Get());
            cl->SetComputeRootDescriptorTable(0, gpu_h(vr_deriv2_heap[i].Get(), 0, inc));
            cl->SetComputeRootDescriptorTable(1, gpu_h(vr_deriv2_heap[i].Get(), 2, inc));
            cl->SetComputeRoot32BitConstants(2, 2, wh, 0);
            for (UINT k = 5; k < 8; ++k)
                trans(cl, l.vr_deriv[k], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cl->Dispatch((l.width + 7) / 8, (l.height + 7) / 8, 1);
            for (UINT k = 5; k < 8; ++k)
                trans(cl, l.vr_deriv[k], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            struct VrParams {
                UINT width, height;
                float alpha, delta, gamma, epsilon;
            };
            const VrParams vp{l.width, l.height, 20.0f, 5.0f, 10.0f, 0.01f};
            for (UINT fixed = 0; fixed < 5; ++fixed) {
                for (auto &c : l.vr_coeff) trans(cl, c, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                set_heap(cl, vr_system_heap[i].Get());
                cl->SetComputeRootSignature(vr_system_root.Get());
                cl->SetPipelineState(vr_system_pso.Get());
                cl->SetComputeRootDescriptorTable(0, gpu_h(vr_system_heap[i].Get(), 0, inc));
                cl->SetComputeRootDescriptorTable(1, gpu_h(vr_system_heap[i].Get(), 11, inc));
                cl->SetComputeRoot32BitConstants(2, 6, &vp, 0);
                cl->Dispatch((l.width + 7) / 8, (l.height + 7) / 8, 1);
                for (auto &c : l.vr_coeff)
                    trans(cl, c, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

                trans(cl, l.vr_dflow_alt, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                bool use_primary = true;
                for (UINT sor = 0; sor < 5; ++sor) {
                    for (UINT parity = 0; parity < 2; ++parity) {
                        auto &heap = use_primary ? vr_sor_heap[i] : vr_sor_alt_heap[i];
                        set_heap(cl, heap.Get());
                        cl->SetComputeRootSignature(vr_sor_root.Get());
                        cl->SetPipelineState(vr_sor_pso.Get());
                        cl->SetComputeRootDescriptorTable(0, gpu_h(heap.Get(), 0, inc));
                        cl->SetComputeRootDescriptorTable(1, gpu_h(heap.Get(), 7, inc));
                        const UINT sc[4] = {l.width, l.height, 0x3FCCCCCDu, parity};
                        cl->SetComputeRoot32BitConstants(2, 4, sc, 0);
                        cl->Dispatch((l.width + 7) / 8, (l.height + 7) / 8, 1);
                        uav_barrier(cl, use_primary ? l.vr_dflow_alt.p.Get()
                                                    : l.vr_dflow.p.Get());
                        if (use_primary) {
                            trans(cl, l.vr_dflow_alt,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            trans(cl, l.vr_dflow, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                        } else {
                            trans(cl, l.vr_dflow,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            trans(cl, l.vr_dflow_alt, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                        }
                        use_primary = !use_primary;
                        ++parity_dispatches;
                    }
                }
                // Ten parity passes leave vr_dflow as the current state.
                uav_barrier(cl, l.vr_dflow.p.Get());
                trans(cl, l.vr_dflow, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

                trans(cl, l.vr_work, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                set_heap(cl, vr_update_heap[i].Get());
                cl->SetComputeRootSignature(vr_update_root.Get());
                cl->SetPipelineState(vr_update_pso.Get());
                cl->SetComputeRootDescriptorTable(0, gpu_h(vr_update_heap[i].Get(), 0, inc));
                cl->SetComputeRootDescriptorTable(1, gpu_h(vr_update_heap[i].Get(), 2, inc));
                cl->SetComputeRoot32BitConstants(2, 2, wh, 0);
                cl->Dispatch((l.width + 7) / 8, (l.height + 7) / 8, 1);
                trans(cl, l.vr_work, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }

            if (i > finest_scale) {
                set_heap(cl, upsample_heap[i].Get());
                cl->SetComputeRootSignature(upsample_root.Get());
                cl->SetPipelineState(upsample_pso.Get());
                const UINT up[4] = {l.width, l.height, levels[i - 1].width,
                                    levels[i - 1].height};
                cl->SetComputeRootDescriptorTable(0, gpu_h(upsample_heap[i].Get(), 0, inc));
                cl->SetComputeRootDescriptorTable(1, gpu_h(upsample_heap[i].Get(), 1, inc));
                cl->SetComputeRoot32BitConstants(2, 4, up, 0);
                trans(cl, levels[i - 1].initial, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                cl->Dispatch((levels[i - 1].width + 7) / 8, (levels[i - 1].height + 7) / 8, 1);
                trans(cl, levels[i - 1].initial,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
        }
    }

    // Final bridge. diag: also emit the OpenCV-contract full-resolution flow
    // into diag_flow and copy it to the readback buffer (acceptance only).
    void record_bridge(ID3D12GraphicsCommandList *cl, bool diag, bool from_cpu_flow) {
        const float out_scale = static_cast<float>(out_w) / static_cast<float>(crop_w);
        const float finest_f = 4.0f;  // 2^finest_scale
        set_heap(cl, misc_heap.Get());
        cl->SetComputeRootSignature(bridge_root.Get());
        cl->SetPipelineState(bridge_pso.Get());
        // t0 = level flow, t1 = cpu flow (contiguous descriptors 8..9)
        cl->SetComputeRootDescriptorTable(0, gpu_h(misc_heap.Get(), desc_level_flow_srv, inc));
        UINT bc[12] = {levels[finest_scale].width, levels[finest_scale].height,
                       crop_w, crop_h, out_w, out_h, 0, 0, from_cpu_flow ? 1u : 0u, 0, 0, 0};
        std::memcpy(bc + 6, &finest_f, sizeof(float));
        std::memcpy(bc + 7, &out_scale, sizeof(float));
        cl->SetComputeRoot32BitConstants(2, 12, bc, 0);
        cl->SetComputeRootDescriptorTable(1, gpu_h(misc_heap.Get(), desc_velocity_uav, inc));
        trans(cl, velocity, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cl->Dispatch((out_w + 7) / 8, (out_h + 7) / 8, 1);
        trans(cl, velocity, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        if (diag) {
            const float one = 1.0f;
            UINT dc[12] = {levels[finest_scale].width, levels[finest_scale].height,
                           crop_w, crop_h, crop_w, crop_h, 0, 0, 0, 0, 0, 0};
            std::memcpy(dc + 6, &finest_f, sizeof(float));
            std::memcpy(dc + 7, &one, sizeof(float));
            set_heap(cl, misc_heap.Get());
            cl->SetComputeRootSignature(bridge_root.Get());
            cl->SetPipelineState(bridge_pso.Get());
            cl->SetComputeRootDescriptorTable(0, gpu_h(misc_heap.Get(), desc_level_flow_srv, inc));
            cl->SetComputeRoot32BitConstants(2, 12, dc, 0);
            cl->SetComputeRootDescriptorTable(1, gpu_h(misc_heap.Get(), desc_diag_flow_uav, inc));
            trans(cl, diag_flow, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cl->Dispatch((crop_w + 7) / 8, (crop_h + 7) / 8, 1);
            trans(cl, diag_flow, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION s{};
            s.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            s.pResource = diag_flow.p.Get();
            D3D12_TEXTURE_COPY_LOCATION d{};
            d.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            d.pResource = diag_flow_rb.r.p.Get();
            d.PlacedFootprint = diag_flow_rb.fp;
            cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
            trans(cl, diag_flow, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            acceptance_readback_bytes += diag_flow_rb.total_bytes;
            ++product_gpu_copies;
        }
    }

    void record_level_dumps(ID3D12GraphicsCommandList *cl) {
        if (!dump_levels) return;
        for (UINT i = 0; i <= coarsest_scale; ++i) {
            struct DumpSrc { R *res; RB *rb; };
            DumpSrc srcs[6] = {{&levels[i].vr_work, &vr_rb[i]},
                               {&levels[i].dense, &dense_rb[i]},
                               {&levels[i].sparse, &sparse_rb[i]},
                               {&levels[i].cur_a, &chain_a_rb[i]},
                               {&levels[i].cur_b, &chain_b_rb[i]},
                               {&levels[i].gradient, &grad_rb[i]}};
            for (auto &s : srcs) {
                trans(cl, *s.res, D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION src{};
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                src.pResource = s.res->p.Get();
                D3D12_TEXTURE_COPY_LOCATION dst{};
                dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.pResource = s.rb->r.p.Get();
                dst.PlacedFootprint = s.rb->fp;
                cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                trans(cl, *s.res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                acceptance_readback_bytes += s.rb->total_bytes;
                ++product_gpu_copies;
            }
        }
    }

    void record_mask(ID3D12GraphicsCommandList *cl, ID3D12Resource *prev_luma,
                     ID3D12Resource *cur_luma) {
        mk_srv(device, misc_heap.Get(), desc_prev_luma_srv, inc, prev_luma,
               DXGI_FORMAT_R32_FLOAT);
        mk_srv(device, misc_heap.Get(), desc_cur_luma_srv, inc, cur_luma,
               DXGI_FORMAT_R32_FLOAT);
        set_heap(cl, misc_heap.Get());
        cl->SetComputeRootSignature(mask_root.Get());
        cl->SetPipelineState(mask_pso.Get());
        const UINT mconst[4] = {crop_w, crop_h, crop_w, crop_h};
        cl->SetComputeRootDescriptorTable(0, gpu_h(misc_heap.Get(), desc_prev_luma_srv, inc));
        cl->SetComputeRootDescriptorTable(1, gpu_h(misc_heap.Get(), desc_mask_uav, inc));
        cl->SetComputeRoot32BitConstants(2, 4, mconst, 0);
        trans(cl, mask, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cl->Dispatch((crop_w + 7) / 8, (crop_h + 7) / 8, 1);
        trans(cl, mask, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    // Acceptance: upload a CPU OpenCV DIS flow (full-res f32x2, row-packed)
    // for the reference endpoint. Counted separately from the product path.
    bool upload_cpu_flow(ID3D12GraphicsCommandList *cl, const uint8_t *packed, UINT64 bytes) {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        device->GetCopyableFootprints(&cpu_flow.p->GetDesc(), 0, 1, 0, &fp, nullptr, nullptr,
                                      nullptr);
        uint8_t *mapped = nullptr;
        const D3D12_RANGE no_read{0, 0};
        if (!p5_ok(cpu_flow_upload.p->Map(0, &no_read, reinterpret_cast<void **>(&mapped)),
                   "map cpu flow upload", err))
            return false;
        for (UINT y = 0; y < crop_h; ++y)
            std::memcpy(mapped + fp.Offset + static_cast<size_t>(y) * fp.Footprint.RowPitch,
                        packed + static_cast<size_t>(y) * crop_w * 8,
                        static_cast<size_t>(crop_w) * 8);
        cpu_flow_upload.p->Unmap(0, nullptr);
        trans(cl, cpu_flow, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION s{};
        s.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        s.pResource = cpu_flow_upload.p.Get();
        s.PlacedFootprint = fp;
        D3D12_TEXTURE_COPY_LOCATION d{};
        d.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        d.pResource = cpu_flow.p.Get();
        cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
        trans(cl, cpu_flow, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        acceptance_upload_bytes += bytes;
        ++product_gpu_copies;
        return true;
    }
};

// ---------------------------------------------------------------------------
// QSV encode bridge, ported from the committed chain contract
// (codex/gpu-block-motion-opt 3bcbc22 src/realtime/vpl_gpu_full_chain_probe.cpp,
// class ChainEncoder): D3D11-owned shared RGBA8 keyed-mutex textures, D3D12
// opens them for XeSS output, oneVPL imports with IMPORT_SHARED|IMPORT_COPY,
// VPP converts RGB4->NV12 in video memory, H.264 QSV encodes.
struct DisEncoder {
    struct SharedSurface {
        ComPtr<ID3D11Texture2D> texture;
        ComPtr<IDXGIKeyedMutex> mutex;
        ComPtr<ID3D12Resource> opened12;
        UINT key = 0;
    };

    ComPtr<ID3D11Device> d3d11;
    ComPtr<ID3D11DeviceContext> d3d11_context;
    mfxLoader loader = nullptr;
    mfxSession session = nullptr;
    mfxMemoryInterface *memory = nullptr;
    std::vector<SharedSurface> surfaces;
    FILE *output = nullptr;
    UINT out_w = 0, out_h = 0;
    UINT imported_count = 0, converted_count = 0, encoded_count = 0;
    UINT pts_mismatch = 0, pts_unknown = 0;
    UINT64 import_copy_count = 0;
    double first_bit_s = -1.0;
    mfxU32 codec_id = MFX_CODEC_AVC;
    mfxU16 gop = 48;
    double fps = 25.0;
    std::chrono::steady_clock::time_point start;
    std::string error;
    bool ready = false;

    ~DisEncoder() {
        if (session) {
            MFXVideoENCODE_Close(session);
            MFXVideoVPP_Close(session);
            MFXClose(session);
        }
        if (loader) MFXUnload(loader);
        if (output) fclose(output);
    }

    static bool set_frame_info(mfxFrameInfo *info, mfxU32 fourcc, UINT width, UINT height,
                               double fps_value) {
        if (!info || width > 16384 || height > 16384) return false;
        memset(info, 0, sizeof(*info));
        info->FourCC = fourcc;
        info->ChromaFormat = MFX_CHROMAFORMAT_YUV420;
        info->PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
        info->FrameRateExtN = static_cast<mfxU32>(fps_value * 1000.0);
        info->FrameRateExtD = 1000;
        info->CropW = static_cast<mfxU16>(width);
        info->CropH = static_cast<mfxU16>(height);
        info->Width = static_cast<mfxU16>((width + 15u) & ~15u);
        info->Height = static_cast<mfxU16>((height + 15u) & ~15u);
        return true;
    }

    bool create_d3d11(const LUID &target_luid) {
        ComPtr<IDXGIFactory6> factory;
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
            error = "encoder_factory";
            return false;
        }
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc)) ||
                memcmp(&desc.AdapterLuid, &target_luid, sizeof(LUID)) != 0)
                continue;
            adapter = candidate;
            break;
        }
        if (!adapter) {
            error = "encoder_adapter_not_found";
            return false;
        }
        constexpr D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1,
                                                D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL selected{};
        HRESULT hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                                       D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels,
                                       static_cast<UINT>(std::size(levels)),
                                       D3D11_SDK_VERSION, &d3d11, &selected,
                                       &d3d11_context);
        if (FAILED(hr)) {
            error = "encoder_d3d11_create";
            return false;
        }
        ComPtr<IDXGIDevice> dxgi;
        ComPtr<IDXGIAdapter> d3d11_adapter;
        DXGI_ADAPTER_DESC a{}, b{};
        const bool same = SUCCEEDED(d3d11.As(&dxgi)) &&
                          SUCCEEDED(dxgi->GetAdapter(&d3d11_adapter)) &&
                          SUCCEEDED(adapter->GetDesc(&a)) &&
                          SUCCEEDED(d3d11_adapter->GetDesc(&b)) &&
                          memcmp(&a.AdapterLuid, &b.AdapterLuid, sizeof(LUID)) == 0;
        if (!same) {
            error = "encoder_luid_mismatch";
            return false;
        }
        ComPtr<ID3D10Multithread> mt;
        if (SUCCEEDED(d3d11_context.As(&mt))) mt->SetMultithreadProtected(TRUE);
        return true;
    }

    bool init_session() {
        loader = MFXLoad();
        if (!loader) {
            error = "encoder_mfxload";
            return false;
        }
        mfxConfig configs[3]{};
        for (auto &config : configs) config = MFXCreateConfig(loader);
        mfxVariant value{};
        value.Type = MFX_VARIANT_TYPE_U32;
        value.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
        MFXSetConfigFilterProperty(configs[0], (mfxU8 *)"mfxImplDescription.Impl", value);
        value.Data.U32 = MFX_ACCEL_MODE_VIA_D3D11;
        MFXSetConfigFilterProperty(configs[1],
                                   (mfxU8 *)"mfxImplDescription.AccelerationMode", value);
        value.Data.U32 = (2u << 16) | 10u;
        MFXSetConfigFilterProperty(configs[2],
                                   (mfxU8 *)"mfxImplDescription.ApiVersion.Version", value);
        if (MFXCreateSession(loader, 0, &session) != MFX_ERR_NONE) {
            error = "encoder_session";
            return false;
        }
        if (MFXVideoCORE_SetHandle(session, MFX_HANDLE_D3D11_DEVICE,
                                   reinterpret_cast<mfxHDL>(d3d11.Get())) != MFX_ERR_NONE ||
            MFXGetMemoryInterface(session, &memory) != MFX_ERR_NONE || !memory) {
            error = "encoder_handle";
            return false;
        }
        return true;
    }

    bool init_components() {
        mfxVideoParam vpp{};
        set_frame_info(&vpp.vpp.In, MFX_FOURCC_BGR4, out_w, out_h, fps);
        set_frame_info(&vpp.vpp.Out, MFX_FOURCC_NV12, out_w, out_h, fps);
        vpp.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;
        vpp.AsyncDepth = 1;
        mfxVideoParam queried = vpp;
        mfxStatus vpp_q = MFXVideoVPP_Query(session, &vpp, &queried);
        mfxStatus vpp_i = MFXVideoVPP_Init(session, &queried);
        if (vpp_q < MFX_ERR_NONE || vpp_i < MFX_ERR_NONE) {
            char text[96]{};
            std::snprintf(text, sizeof(text), "encoder_vpp_init q=%d i=%d",
                          static_cast<int>(vpp_q), static_cast<int>(vpp_i));
            error = text;
            return false;
        }
        mfxVideoParam enc{};
        enc.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY;
        enc.AsyncDepth = 1;
        enc.mfx.CodecId = codec_id;
        enc.mfx.TargetUsage = 4;
        enc.mfx.RateControlMethod = MFX_RATECONTROL_ICQ;
        enc.mfx.ICQQuality = 20;
        enc.mfx.GopRefDist = 1;
        enc.mfx.GopPicSize = gop;
        set_frame_info(&enc.mfx.FrameInfo, MFX_FOURCC_NV12, out_w, out_h, fps);
        queried = enc;
        if (MFXVideoENCODE_Query(session, &enc, &queried) < MFX_ERR_NONE ||
            MFXVideoENCODE_Init(session, &queried) < MFX_ERR_NONE) {
            error = "encoder_encode_init";
            return false;
        }
        return true;
    }

    bool create_shared_surfaces(ID3D12Device *device, UINT count) {
        surfaces.resize(count);
        for (UINT index = 0; index < count; ++index) {
            SharedSurface &surface = surfaces[index];
            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = out_w;
            desc.Height = out_h;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
            desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                             D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
            if (FAILED(d3d11->CreateTexture2D(&desc, nullptr, &surface.texture))) {
                error = "encoder_shared_create";
                return false;
            }
            if (FAILED(surface.texture.As(&surface.mutex))) {
                error = "encoder_keyed_mutex";
                return false;
            }
            ComPtr<IDXGIResource1> resource;
            if (FAILED(surface.texture.As(&resource))) {
                error = "encoder_dxgiresource1";
                return false;
            }
            HANDLE handle = nullptr;
            if (FAILED(resource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &handle))) {
                error = "encoder_create_shared_handle";
                return false;
            }
            const HRESULT hr = device->OpenSharedHandle(handle, IID_PPV_ARGS(&surface.opened12));
            CloseHandle(handle);
            if (FAILED(hr)) {
                error = "encoder_open_shared_handle";
                return false;
            }
            const D3D12_RESOURCE_DESC opened = surface.opened12->GetDesc();
            if (!(opened.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
                error = "encoder_opened_uav_missing";
                return false;
            }
        }
        return true;
    }

    bool init(ID3D12Device *device, const LUID &luid, UINT width, UINT height, UINT slots,
              const char *output_path, mfxU32 codec, mfxU16 gop_value, double fps_value) {
        out_w = width;
        out_h = height;
        codec_id = codec;
        gop = gop_value;
        fps = fps_value;
        start = std::chrono::steady_clock::now();
        if (!create_d3d11(luid) || !init_session() || !init_components() ||
            !create_shared_surfaces(device, slots))
            return false;
        output = fopen(output_path, "wb");
        if (!output) {
            error = "encoder_output_open";
            return false;
        }
        ready = true;
        return true;
    }

    ID3D12Resource *opened(UINT slot_index) {
        return surfaces[slot_index % surfaces.size()].opened12.Get();
    }

    // A5.3 begin contract: advance the keyed counter so the encoder's
    // AcquireSync(key+1) after the D3D12 fence wait is valid.
    bool begin_frame(UINT slot_index) {
        SharedSurface &surface = surfaces[slot_index % surfaces.size()];
        HRESULT hr = surface.mutex->AcquireSync(surface.key, 15000);
        if (hr != S_OK) {
            error = "encoder_begin_acquire";
            return false;
        }
        hr = surface.mutex->ReleaseSync(surface.key + 1);
        if (FAILED(hr)) {
            error = "encoder_begin_release";
            return false;
        }
        return true;
    }

    static bool write_bitstream(const mfxBitstream &bitstream, FILE *out) {
        return !bitstream.DataLength ||
               fwrite(bitstream.Data + bitstream.DataOffset, 1, bitstream.DataLength, out) ==
                   bitstream.DataLength;
    }

    bool encode_slot(UINT slot_index, UINT frame_index) {
        SharedSurface &surface = surfaces[slot_index % surfaces.size()];
        const UINT key = surface.key + 1;
        HRESULT hr = surface.mutex->AcquireSync(key, 15000);
        if (hr != S_OK) {
            error = "encoder_acquire_sync";
            return false;
        }
        mfxSurfaceD3D11Tex2D external{};
        external.SurfaceInterface.Header.SurfaceType = MFX_SURFACE_TYPE_D3D11_TEX2D;
        external.SurfaceInterface.Header.SurfaceFlags =
            MFX_SURFACE_FLAG_IMPORT_SHARED | MFX_SURFACE_FLAG_IMPORT_COPY;
        external.SurfaceInterface.Header.StructSize = sizeof(external);
        external.texture2D = surface.texture.Get();
        mfxFrameSurface1 *input = nullptr;
        mfxStatus status = memory->ImportFrameSurface(
            memory, MFX_SURFACE_COMPONENT_VPP_INPUT, &external.SurfaceInterface.Header, &input);
        if (status != MFX_ERR_NONE || !input) {
            surface.mutex->ReleaseSync(key + 1);
            error = "encoder_import";
            return false;
        }
        ++imported_count;
        ++import_copy_count;
        const mfxU64 timestamp = static_cast<mfxU64>(frame_index) * 90000 /
                                 static_cast<mfxU64>(fps > 1.0 ? fps : 24.0);
        input->Data.TimeStamp = timestamp;
        mfxFrameSurface1 *converted = nullptr;
        status = MFXMemory_GetSurfaceForVPPOut(session, &converted);
        if (status != MFX_ERR_NONE || !converted) {
            input->FrameInterface->Release(input);
            surface.mutex->ReleaseSync(key + 1);
            error = "encoder_vpp_surface";
            return false;
        }
        mfxSyncPoint vpp_sync = nullptr;
        status = MFXVideoVPP_RunFrameVPPAsync(session, input, converted, nullptr, &vpp_sync);
        input->FrameInterface->Release(input);
        if (status != MFX_ERR_NONE || !vpp_sync) {
            converted->FrameInterface->Release(converted);
            surface.mutex->ReleaseSync(key + 1);
            error = "encoder_vpp_run";
            return false;
        }
        if (MFXVideoCORE_SyncOperation(session, vpp_sync, 15000) != MFX_ERR_NONE) {
            converted->FrameInterface->Release(converted);
            surface.mutex->ReleaseSync(key + 1);
            error = "encoder_vpp_sync";
            return false;
        }
        ++converted_count;
        mfxBitstream bitstream{};
        bitstream.MaxLength = 16u << 20;
        bitstream.Data = static_cast<mfxU8 *>(calloc(bitstream.MaxLength, 1));
        mfxSyncPoint enc_sync = nullptr;
        status = bitstream.Data
                     ? MFXVideoENCODE_EncodeFrameAsync(session, nullptr, converted, &bitstream,
                                                       &enc_sync)
                     : MFX_ERR_MEMORY_ALLOC;
        converted->FrameInterface->Release(converted);
        bool ok = status == MFX_ERR_NONE && enc_sync;
        if (ok) {
            status = MFXVideoCORE_SyncOperation(session, enc_sync, 15000);
            ok = status == MFX_ERR_NONE && write_bitstream(bitstream, output);
            if (ok) {
                ++encoded_count;
                if (first_bit_s < 0.0)
                    first_bit_s =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                            .count();
                if (bitstream.TimeStamp == MFX_TIMESTAMP_UNKNOWN) ++pts_unknown;
                else if (bitstream.TimeStamp != timestamp) ++pts_mismatch;
            }
        } else if (status != MFX_ERR_MORE_DATA) {
            ok = false;
            error = "encoder_encode_frame";
        }
        free(bitstream.Data);
        hr = surface.mutex->ReleaseSync(key + 1);
        if (FAILED(hr)) {
            ok = false;
            error = "encoder_release_sync";
        }
        surface.key += 2;
        return ok;
    }

    bool drain() {
        for (UINT i = 0; i < 64; ++i) {
            mfxBitstream bitstream{};
            bitstream.MaxLength = 16u << 20;
            bitstream.Data = static_cast<mfxU8 *>(calloc(bitstream.MaxLength, 1));
            if (!bitstream.Data) return false;
            mfxSyncPoint sync = nullptr;
            mfxStatus status =
                MFXVideoENCODE_EncodeFrameAsync(session, nullptr, nullptr, &bitstream, &sync);
            if (status == MFX_ERR_MORE_DATA) {
                free(bitstream.Data);
                return true;
            }
            if (status != MFX_ERR_NONE || !sync) {
                free(bitstream.Data);
                return false;
            }
            if (MFXVideoCORE_SyncOperation(session, sync, 15000) != MFX_ERR_NONE ||
                !write_bitstream(bitstream, output)) {
                free(bitstream.Data);
                return false;
            }
            ++encoded_count;
            free(bitstream.Data);
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
struct ProbeOptions {
    std::string input;
    std::string codec = "h264";
    std::string shader_dir = "build";
    std::string report;
    std::string xess_quality = "quality";
    int output_width = 0;
    int output_height = 0;
    int adapter = 0;
    int max_frames = 8;
    int loops = 1;                 // bitstream replay count (0 = unbounded)
    double seconds = 0.0;          // wall-time limit (0 = off)
    std::string luma_dir;          // acceptance: decoded Y-plane dump
    int luma_frames = 0;
    std::string flow_dir;          // acceptance: GPU DIS flow dumps
    int flow_every = 0;
    std::string xess_dir;          // acceptance: XeSS output dumps
    int xess_every = 0;
    std::string cpu_flow_prefix;   // acceptance: CPU DIS flow reference endpoint
    std::string encode_output;     // product: QSV encode to elementary stream
    std::string stats_log;         // VRAM/RAM sampling log
    std::string dump_levels_dir;   // acceptance: per-level DIS stage dumps
    bool skip_dis = false;         // with cpu_flow_prefix
    bool skip_xess = false;
    bool no_mask = false;
};

struct FrameStats {
    double wall_ms = 0.0;
    double gpu_ms = -1.0;
    double color_ms = -1.0, dis_ms = -1.0, bridge_mask_ms = -1.0, xess_ms = -1.0;
};

}  // namespace

#ifndef XESS_FG_DIS_EMBED
int main(int argc, char **argv) {
    ProbeOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto take = [&](std::string &v) {
            if (i + 1 >= argc) return false;
            v = argv[++i];
            return true;
        };
        if (arg == "--input" || arg == "-i") take(opt.input);
        else if (arg == "--codec") {
            take(opt.codec);
            if (opt.codec == "avc") opt.codec = "h264";
            if (opt.codec == "h265") opt.codec = "hevc";
        }
        else if (arg == "--shader-dir") take(opt.shader_dir);
        else if (arg == "--report") take(opt.report);
        else if (arg == "--xess-quality") take(opt.xess_quality);
        else if (arg == "--output-width") opt.output_width = std::atoi(argv[++i]);
        else if (arg == "--output-height") opt.output_height = std::atoi(argv[++i]);
        else if (arg == "--adapter") opt.adapter = std::atoi(argv[++i]);
        else if (arg == "--max-frames") opt.max_frames = std::atoi(argv[++i]);
        else if (arg == "--loops") opt.loops = std::atoi(argv[++i]);
        else if (arg == "--seconds") opt.seconds = std::atof(argv[++i]);
        else if (arg == "--luma-dir") take(opt.luma_dir);
        else if (arg == "--luma-frames") opt.luma_frames = std::atoi(argv[++i]);
        else if (arg == "--flow-dir") take(opt.flow_dir);
        else if (arg == "--flow-every") opt.flow_every = std::atoi(argv[++i]);
        else if (arg == "--xess-dir") take(opt.xess_dir);
        else if (arg == "--xess-every") opt.xess_every = std::atoi(argv[++i]);
        else if (arg == "--cpu-flow-prefix") take(opt.cpu_flow_prefix);
        else if (arg == "--encode") take(opt.encode_output);
        else if (arg == "--stats-log") take(opt.stats_log);
        else if (arg == "--skip-dis") opt.skip_dis = true;
        else if (arg == "--skip-xess") opt.skip_xess = true;
        else if (arg == "--no-mask") opt.no_mask = true;
        else if (arg == "--dump-levels-dir") take(opt.dump_levels_dir);
        else {
            std::fprintf(stderr, "unknown arg %s\n", arg.c_str());
            return 2;
        }
    }
    if (opt.input.empty()) {
        std::fprintf(stderr,
                     "usage: vpl-gpu-dis-perfect-probe --input <h264/hevc> [--codec h264|hevc] "
                     "[--max-frames N] [--loops N] [--seconds S] [--adapter N] "
                     "[--output-width W --output-height H] [--xess-quality Q] "
                     "[--encode out.h264] [--luma-dir D --luma-frames N] "
                     "[--flow-dir D --flow-every K] [--xess-dir D --xess-every K] "
                     "[--cpu-flow-prefix P] [--skip-dis] [--skip-xess] [--no-mask] "
                     "[--stats-log PATH] [--report PATH] --shader-dir DIR\n");
        return 2;
    }

    Result result;
    result.input = opt.input;
    result.codec = opt.codec;
    result.requested = opt.max_frames;
    {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
    }
    ComRuntime com;
    if (!com.usable()) return 3;
    std::vector<mfxU8> compressed;
    if (!ReadCompressed(opt.input, compressed)) return 4;

    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D11Device> d3d11;
    ComPtr<ID3D11DeviceContext> d3d11_context;
    ComPtr<ID3D12Device> d3d12;
    if (!CreateDevices(opt.adapter, adapter, d3d11, d3d11_context, d3d12, result)) return 5;
    if (!d3d12 || result.d3d11_luid != result.d3d12_luid) {
        result.block = "adapter_luid_mismatch";
        return 6;
    }
    DXGI_ADAPTER_DESC1 adapter_desc{};
    adapter->GetDesc1(&adapter_desc);
    ComPtr<IDXGIAdapter3> adapter3;
    adapter.As(&adapter3);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    // XeSS D3D12 requires a direct queue (compute-only queues hang the Intel
    // driver when XeSS kernels are submitted — Gate 3 finding).
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(d3d12->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) ||
        FAILED(d3d12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
        result.block = "CreateCommandQueue";
        return 6;
    }
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(d3d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             IID_PPV_ARGS(&allocator))) ||
        FAILED(d3d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                        nullptr, IID_PPV_ARGS(&list)))) {
        result.block = "CreateCommandList";
        return 6;
    }
    // A freshly created list is in the recording state; close it once so the
    // uniform per-frame Reset path is legal.
    list->Close();

    // ---- oneVPL decode session (D3D11 video memory, exportable surfaces) ----
    const mfxU32 codec = opt.codec == "h264" ? MFX_CODEC_AVC : MFX_CODEC_HEVC;
    mfxLoader loader = MFXLoad();
    if (!loader) return 7;
    mfxConfig configs[5]{};
    bool filters_ok = true;
    for (auto &config : configs) config = MFXCreateConfig(loader);
    filters_ok =
        filters_ok &&
        SetFilter(configs[0], "mfxImplDescription.Impl", MFX_IMPL_TYPE_HARDWARE) &&
        SetFilter(configs[1], "mfxImplDescription.ApiVersion.Version", kApiVersion) &&
        SetFilter(configs[2], "mfxImplDescription.AccelerationMode", MFX_ACCEL_MODE_VIA_D3D11) &&
        SetFilter(configs[3], "mfxImplDescription.mfxDecoderDescription.decoder.CodecID", codec) &&
        SetFilter(configs[4], "mfxSurfaceTypesSupported.surftype.SurfaceType",
                  MFX_SURFACE_TYPE_D3D11_TEX2D) &&
        SetFilter(configs[4], "mfxSurfaceTypesSupported.surftype.surfcomp.SurfaceComponent",
                  MFX_SURFACE_COMPONENT_DECODE) &&
        SetFilter(configs[4], "mfxSurfaceTypesSupported.surftype.surfcomp.SurfaceFlags",
                  MFX_SURFACE_FLAG_EXPORT_SHARED);
    if (!filters_ok) return 8;
    mfxSession session = nullptr;
    if (MFXCreateSession(loader, 0, &session) != MFX_ERR_NONE || !session) return 9;
    if (MFXVideoCORE_SetHandle(session, MFX_HANDLE_D3D11_DEVICE,
                               reinterpret_cast<mfxHDL>(d3d11.Get())) != MFX_ERR_NONE)
        return 10;
    mfxBitstream bitstream{};
    bitstream.Data = compressed.data();
    bitstream.MaxLength = static_cast<mfxU32>(compressed.size());
    bitstream.DataLength = static_cast<mfxU32>(compressed.size());
    bitstream.CodecId = codec;
    mfxVideoParam decode_params{};
    decode_params.mfx.CodecId = codec;
    decode_params.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    if (MFXVideoDECODE_DecodeHeader(session, &bitstream, &decode_params) != MFX_ERR_NONE ||
        MFXVideoDECODE_Init(session, &decode_params) < MFX_ERR_NONE)
        return 11;
    result.width = decode_params.mfx.FrameInfo.Width;
    result.height = decode_params.mfx.FrameInfo.Height;
    result.crop_w = decode_params.mfx.FrameInfo.CropW;
    result.crop_h = decode_params.mfx.FrameInfo.CropH;
    result.fourcc = FourCCName(decode_params.mfx.FrameInfo.FourCC);
    const int crop_w = result.crop_w ? result.crop_w : result.width;
    const int crop_h = result.crop_h ? result.crop_h : result.height;
    const int tex_w = result.width;
    const int tex_h = result.height;
    const int out_w = opt.output_width > 0 ? opt.output_width : crop_w * 2;
    const int out_h = opt.output_height > 0 ? opt.output_height : crop_h * 2;
    std::printf("p5_probe input=%s codec=%s decode_tex=%ux%u crop=%dx%d output=%dx%d "
                "compressed_bytes=%zu\n",
                opt.input.c_str(), opt.codec.c_str(), tex_w, tex_h, crop_w, crop_h, out_w,
                out_h, compressed.size());

    // ---- DIS graph ----
    DisPipeline dis;
    dis.dump_levels = !opt.dump_levels_dir.empty();
    if (!dis.init(d3d12.Get(), opt.shader_dir, static_cast<UINT>(crop_w),
                  static_cast<UINT>(crop_h), static_cast<UINT>(out_w),
                  static_cast<UINT>(out_h))) {
        std::fprintf(stderr, "dis_init_failed %s\n", dis.err.c_str());
        result.block = dis.err;
        return 12;
    }
    std::printf("dis_graph=init crop=%ux%u finest=%u coarsest=%u persistent_pools=1 "
                "product_cpu_full_frame_boundary=0\n",
                crop_w, crop_h, dis.finest_scale, dis.coarsest_scale);

    // ---- XeSS ----
    xess_context_handle_t xess = nullptr;
    bool xess_ready = false;
    std::string xess_error = "none";
    if (!opt.skip_xess) {
        if (xessD3D12CreateContext(d3d12.Get(), &xess) != XESS_RESULT_SUCCESS || !xess) {
            xess_error = "xessD3D12CreateContext";
        } else {
            xess_d3d12_init_params_t params{};
            params.outputResolution = {static_cast<uint32_t>(out_w),
                                       static_cast<uint32_t>(out_h)};
            params.qualitySetting =
                opt.xess_quality == "performance" ? XESS_QUALITY_SETTING_PERFORMANCE
                : opt.xess_quality == "balanced"  ? XESS_QUALITY_SETTING_BALANCED
                : opt.xess_quality == "quality"   ? XESS_QUALITY_SETTING_QUALITY
                                                  : XESS_QUALITY_SETTING_ULTRA_QUALITY;
            params.initFlags = XESS_INIT_FLAG_LDR_INPUT_COLOR | XESS_INIT_FLAG_HIGH_RES_MV;
            if (!opt.no_mask) params.initFlags |= XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK;
            if (xessD3D12BuildPipelines(xess, nullptr, true, params.initFlags) !=
                XESS_RESULT_SUCCESS) {
                xess_error = "xessD3D12BuildPipelines";
            } else if (xessD3D12Init(xess, &params) != XESS_RESULT_SUCCESS) {
                xess_error = "xessD3D12Init";
            } else if (xessSetVelocityScale(xess, 1.0f, 1.0f) != XESS_RESULT_SUCCESS) {
                xess_error = "xessSetVelocityScale";
            } else if (!opt.no_mask && xessSetMaxResponsiveMaskValue(xess, 0.8f) !=
                                           XESS_RESULT_SUCCESS) {
                xess_error = "xessSetMaxResponsiveMaskValue";
            } else {
                xess_ready = true;
            }
        }
        if (!xess_ready) result.block = xess_error;
    } else {
        xess_error = "skipped";
    }
    std::printf("xess_init=%d quality=%s flags=LDR|HIGH_RES_MV%s error=%s\n",
                xess_ready ? 1 : 0, opt.xess_quality.c_str(), opt.no_mask ? "" : "|RESPONSIVE_MASK",
                xess_error.c_str());

    // ---- encode bridge ----
    DisEncoder encoder;
    const bool encode_on = !opt.encode_output.empty();
    if (encode_on &&
        !encoder.init(d3d12.Get(), adapter_desc.AdapterLuid, static_cast<UINT>(out_w),
                      static_cast<UINT>(out_h), 2, opt.encode_output.c_str(), MFX_CODEC_AVC,
                      48, 25.0)) {
        std::fprintf(stderr, "encoder_init_failed %s\n", encoder.error.c_str());
        result.block = encoder.error;
        return 12;
    }

    // ---- XeSS output slots (shared when encoding, plain otherwise) ----
    R xess_out[2];
    RB xess_dump_rb[2];
    bool have_xess_dump = false;
    if (encode_on) {
        for (UINT s = 0; s < 2; ++s) {
            xess_out[s].p = encoder.opened(s);
            xess_out[s].s = D3D12_RESOURCE_STATE_COMMON;
        }
    } else {
        for (UINT s = 0; s < 2; ++s) {
            if (!make_tex(d3d12.Get(), static_cast<UINT>(out_w), static_cast<UINT>(out_h),
                          DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_STATE_COMMON,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, xess_out[s], dis.err))
                return 12;
        }
    }
    if (!opt.xess_dir.empty()) {
        for (UINT s = 0; s < 2; ++s) {
            if (!make_readback_buf(d3d12.Get(), xess_out[s].p.Get(), xess_dump_rb[s], dis.err))
                return 12;
        }
        have_xess_dump = true;
    }

    // D3D11 staging texture for the decoded-luma acceptance dump.
    ComPtr<ID3D11Texture2D> staging_nv12;
    if (!opt.luma_dir.empty()) {
        D3D11_TEXTURE2D_DESC sd{};
        sd.Width = static_cast<UINT>(tex_w);
        sd.Height = static_cast<UINT>(tex_h);
        sd.MipLevels = 1;
        sd.ArraySize = 1;
        sd.Format = DXGI_FORMAT_NV12;
        sd.SampleDesc.Count = 1;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(d3d11->CreateTexture2D(&sd, nullptr, &staging_nv12))) {
            result.block = "staging_nv12_create";
            return 12;
        }
    }

    FILE *stats = nullptr;
    if (!opt.stats_log.empty()) stats = fopen(opt.stats_log.c_str(), "w");
    UINT64 vram_min = 0, vram_max = 0;
    auto vram_line = [&](UINT64 frame_index) {
        if (!stats) return;
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (adapter3 && SUCCEEDED(adapter3->QueryVideoMemoryInfo(
                            0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            PROCESS_MEMORY_COUNTERS pmc{};
            GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
            std::fprintf(stats,
                         "frame=%llu local_usage=%llu local_budget=%llu rss=%llu\n",
                         static_cast<unsigned long long>(frame_index),
                         static_cast<unsigned long long>(info.CurrentUsage),
                         static_cast<unsigned long long>(info.Budget),
                         static_cast<unsigned long long>(pmc.WorkingSetSize));
            if (info.CurrentUsage > vram_max) vram_max = info.CurrentUsage;
            if (vram_min == 0 || info.CurrentUsage < vram_min) vram_min = info.CurrentUsage;
        }
        fflush(stats);
    };

    // ---- frame loop ----
    std::vector<FrameStats> frame_stats;
    std::vector<std::pair<UINT, UINT64>> xess_dump_pending;
    ComPtr<ID3D12Resource> imported_prev;
    mfxFrameSurface1 *pending_surface = nullptr;
    UINT64 global_frame = 0;
    int processed_this_run = 0;
    int loops_done = 0;

    bool blocked = false;
    const auto run_start = std::chrono::steady_clock::now();

    for (int loop = 0; (loop < opt.loops || opt.loops == 0) && !blocked; ++loop) {
        bitstream.DataOffset = 0;
        bitstream.DataLength = static_cast<mfxU32>(compressed.size());
        bool draining = false;
        int drain_idle = 0;
        bool loop_done = false;
        while (!loop_done && !blocked) {
            if (opt.seconds > 0.0) {
                const double elapsed = std::chrono::duration<double>(
                                           std::chrono::steady_clock::now() - run_start)
                                           .count();
                if (elapsed >= opt.seconds) {
                    loops_done = loop;
                    loop_done = true;
                    break;
                }
            }
            if (processed_this_run >= opt.max_frames) {
                loop_done = true;
                break;
            }
            // Release the previous imported surface before the decoder may
            // recycle its D3D11 pool entry (Gate 3 lesson).
            imported_prev.Reset();
            mfxFrameSurface1 *surface = nullptr;
            mfxSyncPoint sync = nullptr;
            const mfxBitstream *input =
                (!draining && bitstream.DataLength > 0) ? &bitstream : nullptr;
            if (!input) draining = true;
            const mfxStatus status = MFXVideoDECODE_DecodeFrameAsync(
                session, const_cast<mfxBitstream *>(input), nullptr, &surface, &sync);
            if (status == MFX_ERR_NONE && surface) {
                drain_idle = 0;
                if (surface->FrameInterface->Synchronize(surface, 5000) != MFX_ERR_NONE) {
                    result.block = "FrameInterface.Synchronize";
                    surface->FrameInterface->Release(surface);
                    blocked = true;
                    break;
                }
                mfxSurfaceHeader header{};
                header.SurfaceType = MFX_SURFACE_TYPE_D3D11_TEX2D;
                header.SurfaceFlags = MFX_SURFACE_FLAG_EXPORT_SHARED;
                mfxSurfaceHeader *exported_header = nullptr;
                const mfxStatus export_status =
                    surface->FrameInterface->Export(surface, header, &exported_header);
                if (export_status != MFX_ERR_NONE || !exported_header) {
                    result.block = "FrameInterface.Export=" + std::to_string(export_status);
                    surface->FrameInterface->Release(surface);
                    blocked = true;
                    break;
                }
                auto *exported = reinterpret_cast<mfxSurfaceD3D11Tex2D *>(exported_header);
                ID3D11Texture2D *native =
                    reinterpret_cast<ID3D11Texture2D *>(exported->texture2D);
                result.exported++;
                result.decoded++;

                // acceptance-only luma dump (D3D11 staging readback)
                if (staging_nv12 && static_cast<int>(global_frame) < opt.luma_frames) {
                    d3d11_context->CopySubresourceRegion(staging_nv12.Get(), 0, 0, 0, 0,
                                                         native, 0, nullptr);
                    D3D11_MAPPED_SUBRESOURCE mapped{};
                    if (SUCCEEDED(d3d11_context->Map(staging_nv12.Get(), 0, D3D11_MAP_READ, 0,
                                                     &mapped))) {
                        CreateDirectoryA(opt.luma_dir.c_str(), nullptr);
                        char path[MAX_PATH]{};
                        std::snprintf(path, sizeof(path), "%s\\frame_%04u.y",
                                      opt.luma_dir.c_str(), static_cast<unsigned>(global_frame));
                        FILE *f = nullptr;
                        if (!fopen_s(&f, path, "wb") && f) {
                            for (int y = 0; y < crop_h; ++y)
                                fwrite(static_cast<const uint8_t *>(mapped.pData) +
                                           static_cast<size_t>(y) * mapped.RowPitch,
                                       1, static_cast<size_t>(crop_w), f);
                            fclose(f);
                            dis.acceptance_readback_bytes +=
                                static_cast<UINT64>(crop_w) * crop_h;
                        }
                        d3d11_context->Unmap(staging_nv12.Get(), 0);
                    }
                }

                // open the D3D11 surface in D3D12 (zero-copy import)
                ComPtr<IDXGIResource1> resource;
                HANDLE shared = nullptr;
                ComPtr<ID3D12Resource> imported;
                bool imported_ok = false;
                if (SUCCEEDED(native->QueryInterface(IID_PPV_ARGS(&resource))) &&
                    SUCCEEDED(resource->CreateSharedHandle(
                        nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                        nullptr, &shared))) {
                    result.shared_handle_created++;
                    if (SUCCEEDED(d3d12->OpenSharedHandle(shared, IID_PPV_ARGS(&imported)))) {
                        result.d3d12_opened++;
                        imported_ok = true;
                    }
                    CloseHandle(shared);
                }
                exported->SurfaceInterface.Release(&exported->SurfaceInterface);
                // The oneVPL surface stays referenced until this frame's GPU
                // work has completed and been fence-waited (Gate 3 ordering):
                // releasing it earlier lets the decoder pool recycle the NV12
                // texture while the D3D12 list is still reading it.
                pending_surface = surface;
                if (!imported_ok) {
                    result.block = "import_failed";
                    blocked = true;
                    break;
                }
                R imported_r;
                imported_r.p = imported;
                imported_r.s = D3D12_RESOURCE_STATE_COMMON;

                R &luma_cur = (global_frame & 1) ? dis.luma_b : dis.luma_a;
                R &luma_prev = (global_frame & 1) ? dis.luma_a : dis.luma_b;
                const UINT role = (global_frame & 1) ? 1u : 0u;
                const bool frame0_self = global_frame == 0;
                dis.bind_role_descriptors(role, frame0_self, imported.Get());

                FrameStats fs;
                const auto frame_t0 = std::chrono::steady_clock::now();
                D3D12_QUERY_HEAP_DESC qdesc{};
                qdesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
                qdesc.Count = 6;
                ComPtr<ID3D12QueryHeap> query_heap;
                R query_rb;
                bool have_query = false;
                if (SUCCEEDED(d3d12->CreateQueryHeap(&qdesc, IID_PPV_ARGS(&query_heap))) &&
                    make_buf(d3d12.Get(), 48, D3D12_HEAP_TYPE_READBACK,
                             D3D12_RESOURCE_STATE_COPY_DEST, query_rb, dis.err)) {
                    have_query = true;
                }

                // Strictly serial frames: the previous frame's fence completed,
                // so the allocator/list pair can be recycled.
                if (FAILED(allocator->Reset()) ||
                    FAILED(list->Reset(allocator.Get(), nullptr))) {
                    result.block = "command_list_reset";
                    blocked = true;
                    break;
                }
                trans(list.Get(), imported_r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if (have_query) list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
                dis.record_color(list.Get(), imported.Get(), luma_cur);
                if (have_query) list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);

                const bool want_diag = !opt.flow_dir.empty() && opt.flow_every > 0 &&
                                       (global_frame % static_cast<UINT64>(opt.flow_every)) == 0;
                if (!opt.cpu_flow_prefix.empty()) {
                    // Acceptance endpoint: skip DIS, upload the CPU OpenCV DIS
                    // flow for this pair.
                    char path[MAX_PATH]{};
                    std::snprintf(path, sizeof(path), "%s_%04u.f32x2",
                                  opt.cpu_flow_prefix.c_str(),
                                  static_cast<unsigned>(global_frame));
                    std::ifstream in(path, std::ios::binary | std::ios::ate);
                    if (!in) {
                        result.block = std::string("cpu_flow_open=") + path;
                        blocked = true;
                        break;
                    }
                    const std::streamoff sz = in.tellg();
                    std::vector<uint8_t> file_bytes(static_cast<size_t>(sz));
                    in.seekg(0, std::ios::beg);
                    in.read(reinterpret_cast<char *>(file_bytes.data()), sz);
                    if (!dis.upload_cpu_flow(list.Get(), file_bytes.data(), file_bytes.size())) {
                        result.block = dis.err;
                        blocked = true;
                        break;
                    }
                    dis.record_bridge(list.Get(), false, true);
                } else if (!opt.skip_dis) {
                    dis.record_graph(list.Get(), role, frame0_self);
                    dis.record_bridge(list.Get(), want_diag, false);
                    dis.record_level_dumps(list.Get());
                }
                if (have_query) list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
                if (!opt.no_mask && xess_ready)
                    dis.record_mask(list.Get(), luma_prev.p.Get(), luma_cur.p.Get());
                if (have_query) list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);

                const bool want_xess_dump =
                    have_xess_dump && opt.xess_every > 0 &&
                    (global_frame % static_cast<UINT64>(opt.xess_every)) == 0 && xess_ready;
                if (xess_ready) {
                    R &out_slot = xess_out[global_frame & 1];
                    if (encode_on && !encoder.begin_frame(static_cast<UINT>(global_frame & 1))) {
                        result.block = encoder.error;
                        blocked = true;
                        break;
                    }
                    trans(list.Get(), out_slot, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    xess_d3d12_execute_params_t exec{};
                    exec.inputWidth = static_cast<uint32_t>(crop_w);
                    exec.inputHeight = static_cast<uint32_t>(crop_h);
                    exec.jitterOffsetX = 0.0f;
                    exec.jitterOffsetY = 0.0f;
                    exec.exposureScale = 1.0f;
                    exec.resetHistory = global_frame == 0 ? 1 : 0;
                    exec.pColorTexture = dis.color.p.Get();
                    exec.pVelocityTexture = dis.velocity.p.Get();
                    exec.pOutputTexture = out_slot.p.Get();
                    exec.pDepthTexture = nullptr;
                    exec.pExposureScaleTexture = nullptr;
                    exec.pResponsivePixelMaskTexture = opt.no_mask ? nullptr : dis.mask.p.Get();
                    if (xessD3D12Execute(xess, list.Get(), &exec) != XESS_RESULT_SUCCESS) {
                        result.block = "xessD3D12Execute";
                        blocked = true;
                        break;
                    }
                    trans(list.Get(), out_slot, D3D12_RESOURCE_STATE_COMMON);
                    if (want_xess_dump) {
                        trans(list.Get(), out_slot, D3D12_RESOURCE_STATE_COPY_SOURCE);
                        D3D12_TEXTURE_COPY_LOCATION s{};
                        s.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        s.pResource = out_slot.p.Get();
                        D3D12_TEXTURE_COPY_LOCATION d{};
                        d.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        d.pResource = xess_dump_rb[global_frame & 1].r.p.Get();
                        d.PlacedFootprint = xess_dump_rb[global_frame & 1].fp;
                        list->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
                        trans(list.Get(), out_slot, D3D12_RESOURCE_STATE_COMMON);
                        dis.acceptance_readback_bytes +=
                            xess_dump_rb[global_frame & 1].total_bytes;
                        xess_dump_pending.emplace_back(static_cast<UINT>(global_frame & 1),
                                                       global_frame);
                    }
                }
                if (have_query) list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 4);
                trans(list.Get(), imported_r, D3D12_RESOURCE_STATE_COMMON);
                if (have_query) {
                    list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 5);
                    list->ResolveQueryData(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 6,
                                           query_rb.p.Get(), 0);
                }

                if (list->Close() != S_OK) {
                    result.block = "CommandListClose";
                    blocked = true;
                    break;
                }
                ID3D12CommandList *lists[] = {list.Get()};
                queue->ExecuteCommandLists(1, lists);
                if (FAILED(queue->Signal(fence.Get(), global_frame + 2))) {
                    result.block = "queue_signal";
                    blocked = true;
                    break;
                }
                if (!wait_fence(queue.Get(), fence.Get(), global_frame + 2, dis.err)) {
                    result.block = dis.err;
                    blocked = true;
                    break;
                }
                const HRESULT removed = d3d12->GetDeviceRemovedReason();
                // The GPU work (including every read of the imported NV12) has
                // completed; the decoder pool may now recycle the surface.
                if (pending_surface) {
                    pending_surface->FrameInterface->Release(pending_surface);
                    pending_surface = nullptr;
                }
                if (FAILED(removed)) {
                    result.block = "device_removed=" + p5_hr(removed);
                    blocked = true;
                    break;
                }
                const auto frame_t1 = std::chrono::steady_clock::now();
                fs.wall_ms =
                    std::chrono::duration<double, std::milli>(frame_t1 - frame_t0).count();

                if (have_query) {
                    D3D12_RANGE range{0, 48};
                    uint64_t *ts = nullptr;
                    UINT64 freq = 0;
                    if (SUCCEEDED(queue->GetTimestampFrequency(&freq)) && freq &&
                        SUCCEEDED(query_rb.p->Map(0, &range, reinterpret_cast<void **>(&ts)))) {
                        fs.color_ms = static_cast<double>(ts[1] - ts[0]) * 1000.0 / freq;
                        fs.dis_ms = static_cast<double>(ts[2] - ts[1]) * 1000.0 / freq;
                        fs.bridge_mask_ms = static_cast<double>(ts[3] - ts[2]) * 1000.0 / freq;
                        fs.xess_ms = static_cast<double>(ts[4] - ts[3]) * 1000.0 / freq;
                        fs.gpu_ms = static_cast<double>(ts[5] - ts[0]) * 1000.0 / freq;
                        query_rb.p->Unmap(0, nullptr);
                    }
                }

                if (want_diag && opt.cpu_flow_prefix.empty() && !opt.skip_dis) {
                    CreateDirectoryA(opt.flow_dir.c_str(), nullptr);
                    char path[MAX_PATH]{};
                    std::snprintf(path, sizeof(path), "%s\\gpu_flow_%04u.f32x2",
                                  opt.flow_dir.c_str(), static_cast<unsigned>(global_frame));
                    std::string werr;
                    if (!write_packed(dis.diag_flow_rb.r.p.Get(), dis.diag_flow_rb.fp, crop_w,
                                      crop_h, 8, path, werr)) {
                        result.block = werr;
                        blocked = true;
                        break;
                    }
                    if (dis.dump_levels) {
                        for (UINT i = 0; i <= dis.coarsest_scale; ++i) {
                            char lpath[MAX_PATH]{};
                            std::string lerr;
                            const DisLevel &l = dis.levels[i];
                            auto dump1 = [&](RB &rb, const char *fmt, UINT bpp, UINT w2,
                                             UINT h2) {
                                std::snprintf(lpath, sizeof(lpath), fmt, opt.dump_levels_dir.c_str(), i,
                                              static_cast<unsigned>(global_frame));
                                return write_packed(rb.r.p.Get(), rb.fp, w2, h2, bpp, lpath, lerr);
                            };
                            if (i >= dis.finest_scale) {
                                if (!dump1(dis.vr_rb[i], "%s\\s_level_%u_frame_%04u.f32x2", 8,
                                           l.width, l.height) ||
                                    !dump1(dis.dense_rb[i], "%s\\dense_level_%u_frame_%04u.f32x2",
                                           8, l.width, l.height) ||
                                    !dump1(dis.sparse_rb[i],
                                           "%s\\sparse_level_%u_frame_%04u.f32x2", 8,
                                           l.sparse_width, l.sparse_height)) {
                                    result.block = lerr;
                                    blocked = true;
                                    break;
                                }
                            }
                            if (!dump1(dis.chain_a_rb[i], "%s\\chainA_level_%u_frame_%04u.f32", 4,
                                       l.width, l.height) ||
                                !dump1(dis.chain_b_rb[i], "%s\\chainB_level_%u_frame_%04u.f32", 4,
                                       l.width, l.height) ||
                                !dump1(dis.grad_rb[i], "%s\\grad_level_%u_frame_%04u.f32x2", 8,
                                       l.width, l.height) ||
                                !dump1(dis.tensor_rb[i], "%s\\tensorXX_level_%u_frame_%04u.f32", 4,
                                       l.sparse_width, l.sparse_height)) {
                                result.block = lerr;
                                blocked = true;
                                break;
                            }
                        }
                    }
                }
                for (const auto &pending : xess_dump_pending) {
                    CreateDirectoryA(opt.xess_dir.c_str(), nullptr);
                    char path[MAX_PATH]{};
                    std::snprintf(path, sizeof(path), "%s\\out_%04u.rgba", opt.xess_dir.c_str(),
                                  static_cast<unsigned>(pending.second));
                    std::string werr;
                    if (!write_packed(xess_dump_rb[pending.first].r.p.Get(),
                                      xess_dump_rb[pending.first].fp, out_w, out_h, 4, path,
                                      werr)) {
                        result.block = werr;
                        blocked = true;
                        break;
                    }
                }
                xess_dump_pending.clear();

                if (encode_on) {
                    if (!encoder.encode_slot(static_cast<UINT>(global_frame & 1),
                                             static_cast<UINT>(global_frame))) {
                        result.block = encoder.error;
                        blocked = true;
                        break;
                    }
                }

                imported_prev = imported;
                std::fprintf(stderr,
                             "frame=%llu wall=%.2fms gpu=%.2fms color=%.3fms dis=%.3fms "
                             "bridge_mask=%.3fms xess=%.3fms\n",
                             static_cast<unsigned long long>(global_frame), fs.wall_ms,
                             fs.gpu_ms, fs.color_ms, fs.dis_ms, fs.bridge_mask_ms, fs.xess_ms);
                frame_stats.push_back(fs);
                ++global_frame;
                ++processed_this_run;
                if ((global_frame & 31) == 0) vram_line(global_frame);
            } else if (status == MFX_ERR_MORE_DATA) {
                if (draining) {
                    if (++drain_idle >= 8) {
                        loop_done = true;
                        loops_done = loop + 1;
                    }
                    continue;
                }
                draining = true;
            } else if (status == MFX_WRN_DEVICE_BUSY) {
                continue;
            } else if (status < MFX_ERR_NONE) {
                result.block = "DecodeFrameAsync=" + std::to_string(status);
                blocked = true;
                break;
            }
        }
        if (blocked) break;
        if (opt.seconds > 0.0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count() >=
                opt.seconds)
            break;
        if (processed_this_run >= opt.max_frames) break;
    }

    if (stats) {
        vram_line(global_frame);
        fclose(stats);
    }
    if (encode_on) encoder.drain();

    const double wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
    std::vector<double> walls;
    for (size_t i = 8; i < frame_stats.size(); ++i) walls.push_back(frame_stats[i].wall_ms);
    std::sort(walls.begin(), walls.end());
    auto pct = [&](double p) -> double {
        if (walls.empty()) return -1.0;
        const double pos = p * static_cast<double>(walls.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = static_cast<size_t>(std::ceil(pos));
        return walls[lo] + (walls[hi] - walls[lo]) * (pos - lo);
    };
    std::printf(
        "p5_summary frames=%llu loops=%d wall_s=%.3f first_frame_ms=%.2f p50_ms=%.2f "
        "p95_ms=%.2f gpu_copies=%llu acceptance_readback_bytes=%llu "
        "acceptance_upload_bytes=%llu encode_encoded=%u encode_first_bit_s=%.3f "
        "pts_mismatch=%u pts_unknown=%u block=%s\n",
        static_cast<unsigned long long>(global_frame), loops_done, wall_s,
        frame_stats.empty() ? -1.0 : frame_stats.front().wall_ms, pct(0.50), pct(0.95),
        static_cast<unsigned long long>(dis.product_gpu_copies),
        static_cast<unsigned long long>(dis.acceptance_readback_bytes),
        static_cast<unsigned long long>(dis.acceptance_upload_bytes), encoder.encoded_count,
        encoder.first_bit_s, encoder.pts_mismatch, encoder.pts_unknown,
        result.block.empty() ? "none" : result.block.c_str());

    if (!opt.report.empty()) {
        std::ofstream out(opt.report, std::ios::binary | std::ios::trunc);
        if (out) {
            out << "{\n";
            out << "  \"schema\": \"gpu-dis-perfect/p5-probe@1\",\n";
            out << "  \"input\": ";
            JsonString(out, opt.input);
            out << ",\n  \"codec\": ";
            JsonString(out, opt.codec);
            out << ",\n  \"adapter_name\": ";
            JsonString(out, result.adapter_name);
            out << ",\n  \"adapter_luid\": ";
            JsonString(out, result.adapter_luid);
            out << ",\n  \"d3d11_luid\": ";
            JsonString(out, result.d3d11_luid);
            out << ",\n  \"d3d12_luid\": ";
            JsonString(out, result.d3d12_luid);
            out << ",\n  \"same_luid\": " << (result.d3d11_luid == result.d3d12_luid ? 1 : 0)
                << ",\n";
            out << "  \"decode\": {\"width\": " << result.width << ", \"height\": "
                << result.height << ", \"crop_w\": " << crop_w << ", \"crop_h\": " << crop_h
                << ", \"fourcc\": ";
            JsonString(out, result.fourcc);
            out << ", \"iopattern\": \"VIDEO_MEMORY\", \"surface_type\": \"D3D11_TEX2D\", "
                   "\"export_flags\": \"MFX_SURFACE_FLAG_EXPORT_SHARED\"},\n";
            out << "  \"dis\": {\"finest_scale\": " << dis.finest_scale
                << ", \"coarsest_scale\": " << dis.coarsest_scale
                << ", \"parity_dispatches_total\": " << dis.parity_dispatches
                << ", \"patch_dispatches_total\": " << dis.patch_dispatches
                << ", \"product_cpu_upload_bytes\": 0, \"product_cpu_readback_bytes\": 0},\n";
            out << "  \"xess\": {\"ready\": " << (xess_ready ? 1 : 0) << ", \"quality\": ";
            JsonString(out, opt.xess_quality);
            out << ", \"output_w\": " << out_w << ", \"output_h\": " << out_h << "},\n";
            out << "  \"frames\": {\"processed\": " << global_frame << ", \"loops\": "
                << loops_done << ", \"decoded\": " << result.decoded
                << ", \"exported\": " << result.exported
                << ", \"d3d12_opened\": " << result.d3d12_opened << "},\n";
            out << "  \"boundaries\": {\"d3d12_gpu_copies\": " << dis.product_gpu_copies
                << ", \"full_frame_cpu_upload_bytes_product\": 0, "
                   "\"full_frame_cpu_readback_bytes_product\": 0, "
                   "\"acceptance_readback_bytes\": "
                << dis.acceptance_readback_bytes << ", \"acceptance_upload_bytes\": "
                << dis.acceptance_upload_bytes << "},\n";
            out << "  \"timing\": {\"wall_s\": " << wall_s << ", \"first_frame_ms\": "
                << (frame_stats.empty() ? -1.0 : frame_stats.front().wall_ms)
                << ", \"p50_ms\": " << pct(0.50) << ", \"p95_ms\": " << pct(0.95) << "},\n";
            if (encode_on) {
                out << "  \"encode\": {\"encoded\": " << encoder.encoded_count
                    << ", \"imported\": " << encoder.imported_count
                    << ", \"converted\": " << encoder.converted_count
                    << ", \"import_copy_count\": " << encoder.import_copy_count
                    << ", \"first_bit_s\": " << encoder.first_bit_s
                    << ", \"pts_mismatch\": " << encoder.pts_mismatch
                    << ", \"pts_unknown\": " << encoder.pts_unknown << ", \"output\": ";
                JsonString(out, opt.encode_output);
                out << "},\n";
            }
            out << "  \"vram_min_bytes\": " << vram_min << ", \"vram_max_bytes\": " << vram_max
                << ",\n  \"block\": ";
            JsonString(out, result.block);
            out << "\n}\n";
        }
    }
    {
        ComPtr<ID3D12InfoQueue> iq;
        if (SUCCEEDED(d3d12->QueryInterface(IID_PPV_ARGS(&iq)))) {
            const UINT64 n = iq->GetNumStoredMessages();
            for (UINT64 i = 0; i < n && i < 60; ++i) {
                SIZE_T len = 0;
                if (FAILED(iq->GetMessageA(i, nullptr, &len))) continue;
                std::vector<char> buf(len);
                auto *msg = reinterpret_cast<D3D12_MESSAGE *>(buf.data());
                if (SUCCEEDED(iq->GetMessageA(i, msg, &len)) && msg->pDescription)
                    std::fprintf(stderr, "d3d12_debug[%llu] %s\n",
                                 static_cast<unsigned long long>(i), msg->pDescription);
            }
        }
    }
    if (xess) xessDestroyContext(xess);
    MFXVideoDECODE_Close(session);
    MFXClose(session);
    MFXUnload(loader);
    const bool pass = !blocked && global_frame > 0 && result.block.empty();
    return pass ? 0 : 13;
}
#endif  // XESS_FG_DIS_EMBED
