#ifndef XVE_GPU_BLOCK_CORE_H
#define XVE_GPU_BLOCK_CORE_H

/* Source-level adapter for the reviewed GPU Block Full H2 path.
 *
 * This header is deliberately small and header-only.  It gives offline and
 * realtime workers the same implementation identity and frame-pair contract
 * without introducing a DLL, queue, staging resource, or per-frame IPC. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "xve_types.h"
#include "xve_instrumentation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum XveGpuBlockImplementation {
    XVE_GPU_BLOCK_IMPL_LEGACY = 0,
    XVE_GPU_BLOCK_IMPL_FULL_H2 = 1,
    XVE_GPU_BLOCK_IMPL_LITE_V4 = 2
} XveGpuBlockImplementation;

typedef struct XveGpuBlockFramePair {
    XveStructHeader header;
    XveDeviceContract device;
    const XveGpuFrame *previous_luma;
    const XveGpuFrame *current_luma;
    XveGpuFrame *velocity;
    XveGpuFrame *confidence;
    XveGpuFrame *mask;
    void *command_list;
    uint64_t pair_id;
    XveInstrumentation *instrumentation;
} XveGpuBlockFramePair;

/* Environment override is read once at worker initialization.  The default
 * is Full H2; Lite V4 is opt-in until its full Core gate is complete.
 * XVE_GPU_BLOCK_IMPL=legacy exists solely for A/B and rollback. */
static inline XveGpuBlockImplementation xveGpuBlockSelectImplementation(void) {
    const char *value = getenv("XVE_GPU_BLOCK_IMPL");
    if (value && (strcmp(value, "legacy") == 0 ||
                 strcmp(value, "h0") == 0))
        return XVE_GPU_BLOCK_IMPL_LEGACY;
    if (value && (strcmp(value, "lite") == 0 ||
                  strcmp(value, "lite_v4") == 0))
        return XVE_GPU_BLOCK_IMPL_LITE_V4;
    return XVE_GPU_BLOCK_IMPL_FULL_H2;
}

static inline const char *xveGpuBlockMotionShader(
    XveGpuBlockImplementation implementation) {
    return implementation == XVE_GPU_BLOCK_IMPL_LEGACY
        ? "surface_gpu_motion" : "surface_gpu_motion_tile";
}

static inline int32_t xveGpuBlockIsLite(
    XveGpuBlockImplementation implementation) {
    return implementation == XVE_GPU_BLOCK_IMPL_LITE_V4;
}

static inline const char *xveGpuBlockImplementationName(
    XveGpuBlockImplementation implementation) {
    return implementation == XVE_GPU_BLOCK_IMPL_LEGACY ? "legacy" :
           implementation == XVE_GPU_BLOCK_IMPL_LITE_V4 ? "lite_v4" : "full_h2";
}

/* Cheap contract validation at the frame-pair boundary.  It never maps or
 * copies a resource.  Return 0 for a malformed/mixed-device pair. */
static inline int32_t xveGpuBlockValidateFramePair(
    const XveGpuBlockFramePair *pair) {
    if (!pair || !pair->previous_luma || !pair->current_luma ||
        !pair->velocity || !pair->command_list) return 0;
    const XveGpuFrame *previous = pair->previous_luma;
    const XveGpuFrame *current = pair->current_luma;
    if (!previous->resource || !current->resource ||
        previous->device.device != current->device.device ||
        previous->device.queue != current->device.queue ||
        pair->device.device != current->device.device ||
        pair->device.queue != current->device.queue)
        return 0;
    if (previous->width != current->width || previous->height != current->height)
        return 0;
    return 1;
}

/* Native worker convenience form.  This is the same frame-pair contract with
 * metadata already owned by the surrounding worker.  It is intentionally
 * pointer/shape-only and is called once before the two H2 dispatches. */
static inline int32_t xveGpuBlockValidateNativeFramePair(
    void *device, void *queue, void *command_list,
    const void *previous_resource, const void *current_resource,
    uint32_t width, uint32_t height, uint32_t previous_width,
    uint32_t previous_height) {
    if (!device || !queue || !command_list || !previous_resource ||
        !current_resource || width == 0 || height == 0 ||
        width != previous_width || height != previous_height) return 0;
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* XVE_GPU_BLOCK_CORE_H */
