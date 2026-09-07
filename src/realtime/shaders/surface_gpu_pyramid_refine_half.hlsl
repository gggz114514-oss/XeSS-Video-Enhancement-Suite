// H3 Fast half-resolution refinement.  Quarter-level flow is upscaled by 2
// and searched only in a radius-4 window at half resolution.
Texture2D<float> PreviousHalf : register(t0);
Texture2D<float> CurrentHalf : register(t1);
Texture2D<float2> ForwardQuarter : register(t2);
Texture2D<float2> BackwardQuarter : register(t3);
RWTexture2D<float2> ForwardHalf : register(u0);
RWTexture2D<float2> BackwardHalf : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint baseWidth; uint baseHeight;
};

int2 cp(int2 p) {
    return clamp(p, int2(0, 0), int2(int(width) - 1, int(height) - 1));
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
              Texture2D<float2> coarse, int2 p) {
    const int2 bp = min(p / 2, int2(int(baseWidth) - 1, int(baseHeight) - 1));
    const int2 center = int2(round(coarse.Load(int3(bp, 0)) * 2.0f));
    float best = 1e20f;
    int2 best_q = p + center;
    int best_norm = 0x7fffffff;
    [unroll] for (int dy = -4; dy <= 4; ++dy)
        [unroll] for (int dx = -4; dx <= 4; ++dx) {
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
    ForwardHalf[p] = refine(CurrentHalf, PreviousHalf, ForwardQuarter, p);
    BackwardHalf[p] = refine(PreviousHalf, CurrentHalf, BackwardQuarter, p);
}
