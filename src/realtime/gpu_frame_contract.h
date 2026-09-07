#pragma once

// Native GPU provider contract v1. No SDK-specific resource tags or CPU pixel
// arrays cross this boundary. Recording never submits or waits the caller queue.
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <string>
#include <vector>

namespace xess_gpu {
using Microsoft::WRL::ComPtr;
constexpr uint32_t kContractVersion = 1;
constexpr uint64_t kNoFrame = UINT64_MAX;

struct Rational { int64_t numerator = 0; int64_t denominator = 1; };
struct Extent { uint32_t width = 0, height = 0; };
struct Crop { uint32_t x = 0, y = 0, width = 0, height = 0; };
enum class Matrix : uint8_t { Unknown, Rgb, Bt601, Bt709, Bt2020 };
enum class Range : uint8_t { Unknown, Full, Limited };
enum class Transfer : uint8_t { Unknown, Linear, Srgb, Bt709, Pq, Hlg };
enum class ResetReason : uint32_t {
    None = 0, FirstFrame = 1, SceneCut = 2, InputDrop = 4,
    GeometryChange = 8, TimestampDiscontinuity = 16, DeviceChange = 32,
    Explicit = 64
};
enum class Consumer : uint8_t { Motion, Depth, Sr, Fg, Post, Encode, Display };

struct FencePoint {
    ComPtr<ID3D12Fence> fence;
    uint64_t value = 0;
    bool complete() const noexcept {
        return !value || (fence && fence->GetCompletedValue() != UINT64_MAX &&
                         fence->GetCompletedValue() >= value);
    }
};
struct ConsumerRelease { Consumer consumer; FencePoint completion; };

// Owner pins both the texture and any producer pool lease. A decoder's surface
// lease must remain live until every consumer has actually completed, including
// use as the next pair's previous frame. CPU-ready does not mean releasable.
struct TextureView {
    ComPtr<ID3D12Resource> resource;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    Extent allocation;
    Crop valid;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};
struct FrameMetadata {
    uint64_t source_frame_id = kNoFrame;
    uint64_t previous_source_frame_id = kNoFrame;
    uint64_t next_source_frame_id = kNoFrame;
    int64_t original_pts = 0;
    Rational original_time_base;
    Rational duration;
    bool pts_known = false;
    bool pts_synthesized = false;
    Matrix matrix = Matrix::Unknown;
    Range range = Range::Unknown;
    Transfer transfer = Transfer::Unknown;
    LUID adapter_luid{};
    ResetReason reset = ResetReason::None;
};
struct FrameLease {
    FrameMetadata metadata;
    TextureView color;
    ComPtr<IUnknown> producer_pool_owner;
    ComPtr<ID3D12CommandQueue> producer_queue;
    FencePoint produced;
    std::vector<ConsumerRelease> consumers;
    bool consumers_registered = false;
    bool releasable() const noexcept {
        if (!consumers_registered || !produced.complete()) return false;
        for (const auto& item : consumers)
            if (!item.completion.complete()) return false;
        return true;
    }
};

enum class ConfidenceDefinition : uint8_t {
    Unavailable, BlockMatchPhotometricAndUniqueness,
    ForwardBackwardPhotometricReliability, ProviderDocumented
};
struct MotionPacket {
    uint32_t contract_version = kContractVersion;
    uint64_t previous_source_frame_id = kNoFrame;
    uint64_t current_source_frame_id = kNoFrame;
    FrameMetadata current_metadata;
    // Integer texel p represents source pixel center p+0.5. xy is displacement
    // in SOURCE valid-region pixels. Primary direction is current -> previous.
    TextureView current_to_previous;
    TextureView previous_to_current;
    TextureView confidence;
    TextureView reliability;
    ConfidenceDefinition confidence_definition = ConfidenceDefinition::Unavailable;
    std::string confidence_semantics;
    bool has_reverse = false;
    bool first_frame_self = false;
    FencePoint produced;
};
struct DepthPacket {
    uint64_t current_source_frame_id = kNoFrame;
    TextureView inverse_depth; // normalized [0,1], larger means nearer
    std::string model_sha256;
    std::string preprocessing;
    std::string normalization;
    std::string temporal_stabilization;
    uint32_t cadence = 1;
    bool online = false;
    bool gpu_remote_output = false;
    FencePoint produced;
};
struct Counters {
    uint64_t source_pair_analysis_count = 0;
    uint64_t directional_dispatch_count = 0;
    uint64_t first_frame_self_analysis_count = 0;
    uint64_t sr_consume_count = 0;
    uint64_t fg_consume_count = 0;
    uint64_t motion_geometry_dispatch_count = 0;
    uint64_t depth_inference_count = 0;
    uint64_t depth_buffer_to_texture_count = 0;
    uint64_t gpu_copy_count = 0;
    uint64_t vpp_import_copy_count = 0;
    uint64_t full_frame_cpu_readback_bytes = 0;
    uint64_t full_frame_cpu_upload_bytes = 0;
    uint64_t statistics_readback_bytes = 0;
    bool fg_mask_tagged = false; // public XeFG has no responsive-mask tag
};
struct RecordContext {
    ID3D12Device* device = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    LUID adapter_luid{};
    // The caller signals this fence AFTER submitting the recorded command list.
    FencePoint completion;
    // Optional read-only views of the core's existing converted source planes.
    // Valid only during record(); providers must not retain these pointers.
    // Their source IDs/color metadata are current/previous FrameLease metadata;
    // ownership stays with the frame slots and their registered consumer fences.
    const TextureView* current_rgba = nullptr;
    const TextureView* previous_rgba = nullptr;
};
struct ProviderConfig { Extent source; uint32_t slots = 4; bool bidirectional = true; };
class MotionProvider {
public:
    virtual ~MotionProvider() = default;
    virtual const char* backend_name() const noexcept = 0;
    virtual bool initialize(ID3D12Device*, const ProviderConfig&, std::string&) = 0;
    virtual bool record(const FrameLease* previous, const FrameLease& current,
                        uint32_t slot, RecordContext&, MotionPacket&,
                        Counters&, std::string&) = 0;
};

inline bool same_adapter(LUID a, LUID b) noexcept {
    return a.HighPart == b.HighPart && a.LowPart == b.LowPart;
}
inline bool matches(const FrameLease& frame, const MotionPacket& packet) noexcept {
    return frame.metadata.source_frame_id == packet.current_source_frame_id &&
           frame.metadata.previous_source_frame_id == packet.previous_source_frame_id;
}
// Apply x/y scales independently at the consumer adapter, once. Do not run a
// provider on SR output. Color and adapted packet retain current_source_frame_id.
inline bool geometry_scale(Extent source, Extent target, float& x, float& y) noexcept {
    if (!source.width || !source.height || !target.width || !target.height) return false;
    x = static_cast<float>(target.width) / source.width;
    y = static_cast<float>(target.height) / source.height;
    return true;
}
} // namespace xess_gpu
