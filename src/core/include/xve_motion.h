#ifndef XVE_MOTION_H
#define XVE_MOTION_H

#include <stdint.h>
#include "xve_types.h"
#include "xve_instrumentation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum XveMotionProviderKind {
    XVE_MOTION_GPU_BLOCK_FULL_H2 = 1,
    XVE_MOTION_GPU_BLOCK_LITE_V4 = 2,
    XVE_MOTION_GPU_DIS_EXPERIMENTAL = 3,
    XVE_MOTION_CPU_DIS_ADAPTER = 4
} XveMotionProviderKind;

typedef struct XveMotionProviderDesc {
    XveStructHeader header;
    XveMotionProviderKind kind;
    uint32_t flags;
    XveDeviceContract device;
    XveInstrumentation *instrumentation;
} XveMotionProviderDesc;

typedef struct XveMotionSubmitInfo {
    XveStructHeader header;
    XveGpuFramePair pair;
    XveGpuFrame *velocity_output;
    XveGpuFrame *confidence_output;
    XveGpuFrame *mask_output;
    void *command_list;
    uint32_t reset_history;
    uint32_t reserved;
} XveMotionSubmitInfo;

typedef int32_t (*XveMotionSubmitFn)(void *provider,
                                    const XveMotionSubmitInfo *info,
                                    XveMotionResult *result);

typedef struct XveMotionProviderVTable {
    XveStructHeader header;
    XveMotionSubmitFn submit_frame_pair;
} XveMotionProviderVTable;

#ifdef __cplusplus
}
#endif

#endif /* XVE_MOTION_H */
