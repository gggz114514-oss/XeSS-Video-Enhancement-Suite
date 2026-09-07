#ifndef XVE_CORE_H
#define XVE_CORE_H

#include <stdint.h>
#include "xve_backend.h"
#include "xve_motion.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct XveCoreInfo {
    XveStructHeader header;
    uint32_t abi_major;
    uint32_t abi_minor;
    const char *build_id;
    const char *resource_contract;
} XveCoreInfo;

/* A future shared build may export these symbols from xve-core.dll.  The
 * initial Foundation path keeps them statically linked/inline where that
 * avoids a call/loader boundary. */
uint32_t xveCoreGetAbiVersion(void);
int32_t xveCoreGetInfo(XveCoreInfo *info);

#ifdef __cplusplus
}
#endif

#endif /* XVE_CORE_H */
