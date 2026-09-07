// PIX markers are not loaded in this standalone backend. This preserves AMD's
// original resource/compute code while avoiding the optional PIX SDK header.
#define USE_PIX 1
#define PIX_COLOR(r,g,b) (((r) << 16) | ((g) << 8) | (b))
#include <ffx_dx12.cpp>
