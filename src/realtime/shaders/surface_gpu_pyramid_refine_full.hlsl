// H3 Fast full-resolution refinement.  Half-level flow is upscaled by 2 and
// searched only in a radius-2 window.  No large-radius full-resolution pass.
Texture2D<float> PreviousLuma : register(t0);
Texture2D<float> CurrentLuma : register(t1);
Texture2D<float2> ForwardHalf : register(t2);
Texture2D<float2> BackwardHalf : register(t3);
RWTexture2D<float2> ForwardOut : register(u0);
RWTexture2D<float2> BackwardOut : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
};

int2 cp(int2 p) {
    return clamp(p, int2(0, 0),
                 int2(int(inputWidth) - 1, int(inputHeight) - 1));
}

float patch_cost(Texture2D<float> a, Texture2D<float> b, int2 p, int2 q) {
    float cost = 0.0f;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
        [unroll] for (int dx = -1; dx <= 1; ++dx)
            cost += abs(a.Load(int3(cp(p + int2(dx, dy)), 0)) -
                        b.Load(int3(cp(q + int2(dx, dy)), 0)));
    return cost;
}

float2 refine(Texture2D<float> a, Texture2D<float> b,
              Texture2D<float2> prior, int2 p) {
    const uint base_width = (inputWidth + 1) / 2;
    const uint base_height = (inputHeight + 1) / 2;
    const int2 bp = min(p / 2, int2(int(base_width) - 1, int(base_height) - 1));
    const int2 center = int2(round(prior.Load(int3(bp, 0)) * 2.0f));
    float best = 1e20f;
    int2 best_q = p + center;
    int best_norm = 0x7fffffff;
    [unroll] for (int dy = -2; dy <= 2; ++dy)
        [unroll] for (int dx = -2; dx <= 2; ++dx) {
            const int2 q = p + center + int2(dx, dy);
            const float cost = patch_cost(a, b, p, q);
            const int norm = dx * dx + dy * dy;
            if (cost < best || (cost == best && norm < best_norm)) {
                best = cost;
                best_q = q;
                best_norm = norm;
            }
        }
    return float2(best_q - p);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    const int2 p = int2(tid.xy);
    ForwardOut[p] = refine(CurrentLuma, PreviousLuma, ForwardHalf, p);
    BackwardOut[p] = refine(PreviousLuma, CurrentLuma, BackwardHalf, p);
}
