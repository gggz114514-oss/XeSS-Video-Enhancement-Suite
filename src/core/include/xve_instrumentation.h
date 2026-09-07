#ifndef XVE_INSTRUMENTATION_H
#define XVE_INSTRUMENTATION_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "xve_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct XveBoundaryCounters {
    XveStructHeader header;
    uint64_t gpu_copy_count;
    uint64_t gpu_copy_bytes;
    uint64_t cpu_full_frame_upload_bytes;
    uint64_t cpu_full_frame_readback_bytes;
    uint64_t explicit_barrier_count;
    uint64_t queue_submit_count;
    uint64_t cpu_wait_count;
    uint64_t gpu_timestamp_count;
    uint64_t gpu_timestamp_ns;
} XveBoundaryCounters;

typedef struct XveInstrumentation {
    XveStructHeader header;
    uint8_t enabled;
    uint8_t reserved[7];
    XveBoundaryCounters *counters;
} XveInstrumentation;

static inline void xveInstrumentationInitialize(
    XveInstrumentation *scope, XveBoundaryCounters *counters) {
    if (!scope || !counters) return;
    memset(counters, 0, sizeof(*counters));
    counters->header.size = (uint32_t)sizeof(*counters);
    counters->header.abi_major = XVE_ABI_MAJOR;
    counters->header.abi_minor = XVE_ABI_MINOR;
    memset(scope, 0, sizeof(*scope));
    scope->header.size = (uint32_t)sizeof(*scope);
    scope->header.abi_major = XVE_ABI_MAJOR;
    scope->header.abi_minor = XVE_ABI_MINOR;
    scope->enabled = 0;
    const char *value = getenv("XVE_INSTRUMENT");
    if (value && (strcmp(value, "1") == 0 || strcmp(value, "on") == 0))
        scope->enabled = 1;
    scope->counters = counters;
}

/* These helpers are intentionally branch-only when disabled.  They are safe
 * for a session-local counter block; callers that record from multiple threads
 * should provide their own atomic/merged block. */
static inline void xveInstrumentationAddCopy(XveInstrumentation *scope,
                                             uint64_t bytes) {
    if (!scope || !scope->enabled || !scope->counters) return;
    ++scope->counters->gpu_copy_count;
    scope->counters->gpu_copy_bytes += bytes;
}

static inline void xveInstrumentationAddCpuUpload(XveInstrumentation *scope,
                                                  uint64_t bytes) {
    if (!scope || !scope->enabled || !scope->counters) return;
    scope->counters->cpu_full_frame_upload_bytes += bytes;
}

static inline void xveInstrumentationAddCpuReadback(XveInstrumentation *scope,
                                                    uint64_t bytes) {
    if (!scope || !scope->enabled || !scope->counters) return;
    scope->counters->cpu_full_frame_readback_bytes += bytes;
}

static inline void xveInstrumentationAddBarrier(XveInstrumentation *scope) {
    if (!scope || !scope->enabled || !scope->counters) return;
    ++scope->counters->explicit_barrier_count;
}

static inline void xveInstrumentationAddQueueSubmit(XveInstrumentation *scope) {
    if (!scope || !scope->enabled || !scope->counters) return;
    ++scope->counters->queue_submit_count;
}

static inline void xveInstrumentationAddCpuWait(XveInstrumentation *scope) {
    if (!scope || !scope->enabled || !scope->counters) return;
    ++scope->counters->cpu_wait_count;
}

static inline void xveInstrumentationAddGpuTimestamp(XveInstrumentation *scope,
                                                     uint64_t nanoseconds) {
    if (!scope || !scope->enabled || !scope->counters) return;
    ++scope->counters->gpu_timestamp_count;
    scope->counters->gpu_timestamp_ns += nanoseconds;
}

#ifdef __cplusplus
}
#endif

#endif /* XVE_INSTRUMENTATION_H */
