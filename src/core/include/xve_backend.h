#ifndef XVE_BACKEND_H
#define XVE_BACKEND_H

#include <stdint.h>
#include "xve_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum XveEnhanceBackendKind {
    XVE_BACKEND_XESS_QUALITY = 1,
    XVE_BACKEND_INTEL_MEDIA = 2
} XveEnhanceBackendKind;

typedef struct XveEnhanceBackendDesc {
    XveStructHeader header;
    XveEnhanceBackendKind kind;
    uint32_t operation;
    XveDeviceContract device;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t output_fps_num;
    uint32_t output_fps_den;
} XveEnhanceBackendDesc;

typedef struct XveEnhanceSubmitInfo {
    XveStructHeader header;
    const XveGpuFrame *input;
    XveGpuFrame *output;
    void *command_list;
    uint64_t frame_id;
    int64_t pts;
    uint32_t reset_history;
    uint32_t reserved;
} XveEnhanceSubmitInfo;

typedef int32_t (*XveEnhanceSubmitFn)(void *backend,
                                     const XveEnhanceSubmitInfo *info);

typedef struct XveEnhanceBackendVTable {
    XveStructHeader header;
    XveEnhanceSubmitFn submit_frame;
} XveEnhanceBackendVTable;

#ifdef __cplusplus
}
#endif

#endif /* XVE_BACKEND_H */
