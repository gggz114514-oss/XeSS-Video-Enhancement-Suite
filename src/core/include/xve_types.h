#ifndef XVE_TYPES_H
#define XVE_TYPES_H

/* Stable, opaque C ABI types for the native video path.
 *
 * The ABI intentionally does not expose C++ or D3D12 layout.  A pointer field
 * carries the caller's ID3D12Device/Queue/Resource/Fence directly; no wrapper
 * allocation or pixel copy is implied.  All structures are size-versioned.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XVE_ABI_MAJOR 1u
#define XVE_ABI_MINOR 0u

typedef struct XveStructHeader {
    uint32_t size;
    uint16_t abi_major;
    uint16_t abi_minor;
} XveStructHeader;

typedef enum XvePixelFormat {
    XVE_FORMAT_UNKNOWN = 0,
    XVE_FORMAT_NV12 = 1,
    XVE_FORMAT_RGBA8 = 2,
    XVE_FORMAT_BGRA8 = 3,
    XVE_FORMAT_R16G16_FLOAT = 4,
    XVE_FORMAT_R32_FLOAT = 5,
    XVE_FORMAT_R8_UNORM = 6,
    XVE_FORMAT_P010 = 7
} XvePixelFormat;

typedef enum XveColorSpace {
    XVE_COLOR_UNKNOWN = 0,
    XVE_COLOR_BT601 = 1,
    XVE_COLOR_BT709 = 2,
    XVE_COLOR_BT2020 = 3
} XveColorSpace;

typedef enum XveColorRange {
    XVE_RANGE_UNKNOWN = 0,
    XVE_RANGE_LIMITED = 1,
    XVE_RANGE_FULL = 2
} XveColorRange;

typedef enum XveResourceState {
    XVE_STATE_UNKNOWN = 0,
    XVE_STATE_COMMON = 1,
    XVE_STATE_UAV = 2,
    XVE_STATE_SRV = 3,
    XVE_STATE_COPY_SOURCE = 4,
    XVE_STATE_COPY_DEST = 5,
    XVE_STATE_PRESENT = 6
} XveResourceState;

typedef enum XveOwnership {
    XVE_OWNERSHIP_BORROWED = 0,
    XVE_OWNERSHIP_CALLER = 1,
    XVE_OWNERSHIP_PROVIDER_UNTIL_FENCE = 2
} XveOwnership;

typedef struct XveDeviceContract {
    XveStructHeader header;
    void *device;
    void *queue;
    uint64_t adapter_luid;
    uint32_t queue_type;
    uint32_t reserved;
} XveDeviceContract;

typedef struct XveGpuFrame {
    XveStructHeader header;
    XveDeviceContract device;
    void *resource;
    void *fence;
    uint64_t fence_value;
    uint32_t width;
    uint32_t height;
    uint32_t row_pitch;
    XvePixelFormat format;
    XveColorSpace color_space;
    XveColorRange color_range;
    XveResourceState state;
    XveOwnership ownership;
    uint64_t frame_id;
    int64_t pts;
    void *lifetime_token;
} XveGpuFrame;

typedef struct XveGpuFramePair {
    XveStructHeader header;
    const XveGpuFrame *previous;
    const XveGpuFrame *current;
    uint64_t pair_id;
} XveGpuFramePair;

typedef struct XveMotionResult {
    XveStructHeader header;
    XveGpuFrame velocity;
    XveGpuFrame confidence;
    XveGpuFrame responsive_mask;
    uint32_t direction;
    uint32_t units;
    uint32_t reliable;
    uint32_t reserved;
} XveMotionResult;

#ifdef __cplusplus
}
#endif

#endif /* XVE_TYPES_H */
