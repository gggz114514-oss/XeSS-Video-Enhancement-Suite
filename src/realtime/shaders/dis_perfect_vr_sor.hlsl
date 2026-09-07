// OpenCV 5.0.0 VariationalRefinement red/black SOR solve.
// Dispatches are serialized by parity. Consequently every read below sees
// the latest opposite-colour value, unlike a Jacobi ping-pong approximation.
Texture2D<float> A11 : register(t0);
Texture2D<float> A12 : register(t1);
Texture2D<float> A22 : register(t2);
Texture2D<float> B1 : register(t3);
Texture2D<float> B2 : register(t4);
Texture2D<float> Weights : register(t5);
Texture2D<float2> DFlow : register(t6);
RWTexture2D<float2> DFlowOut : register(u0);
cbuffer Params : register(b0) { uint width; uint height; float omega; uint parity; };
#ifdef DIS_VR_EXACT
#include "dis_cpu_rounding.hlsli"
#else
float cpu_divide(float a, float b) { return a / b; }
#endif
int2 clamp_xy(int2 p) { return clamp(p, int2(0, 0), int2(width - 1, height - 1)); }
float2 d_at(int2 p) { return DFlow.Load(int3(clamp_xy(p), 0)); }
float w_at(int2 p) { return Weights.Load(int3(clamp_xy(p), 0)); }
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    // The host ping-pongs the increment texture on every parity pass. Copy
    // the opposite colour unchanged so the output is a complete state and
    // no SRV/UAV aliasing is needed for in-place checkerboard updates.
    int2 p = int2(id.xy); float2 d = d_at(p), left = d_at(p-int2(1,0)), right = d_at(p+int2(1,0));
    float2 up = d_at(p-int2(0,1)), down = d_at(p+int2(0,1));
    // The host ping-pongs the increment texture on every parity pass. Copy
    // the opposite colour unchanged so the output is a complete state and
    // no SRV/UAV aliasing is needed for in-place checkerboard updates.
    if (((id.x + id.y) & 1u) != parity) { DFlowOut[p] = d; return; }
    float wl = p.x > 0 ? w_at(p-int2(1,0)) : 0.0;
    float wr = p.x + 1 < (int)width ? w_at(p) : 0.0;
    float wu = p.y > 0 ? w_at(p-int2(0,1)) : 0.0;
    float wd = p.y + 1 < (int)height ? w_at(p) : 0.0;
    float su = wl * left.x + wr * right.x + wu * up.x + wd * down.x;
    float sv = wl * left.y + wr * right.y + wu * up.y + wd * down.y;
    float a11 = max(A11.Load(int3(p,0)), 1e-12), a12 = A12.Load(int3(p,0)), a22 = max(A22.Load(int3(p,0)), 1e-12);
    d.x += omega * (cpu_divide(su + B1.Load(int3(p,0)) - d.y * a12, a11) - d.x);
    d.y += omega * (cpu_divide(sv + B2.Load(int3(p,0)) - d.x * a12, a22) - d.y);
    DFlowOut[p] = d;
}
