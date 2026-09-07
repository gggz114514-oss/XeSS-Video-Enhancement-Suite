Texture2D<float> FrameA : register(t0);
Texture2D<float> FrameB : register(t1);
Texture2D<float2> BaseForward : register(t2);
Texture2D<float2> BaseBackward : register(t3);
RWTexture2D<float2> ForwardOut : register(u0);
RWTexture2D<float2> BackwardOut : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint baseWidth; uint baseHeight;
    uint scenario; int shiftX; int shiftY; float angle;
};
int2 cp(int2 p) { return clamp(p, int2(0, 0), int2(width - 1, height - 1)); }
float cost(Texture2D<float> f, Texture2D<float> r, int2 p, int2 q) {
    float total = 0.0f;
    [unroll] for (int py = -1; py <= 1; ++py) [unroll] for (int px = -1; px <= 1; ++px) {
        int2 ap = cp(p + int2(px, py)); int2 bp = cp(q + int2(px, py));
        total += abs(f.Load(int3(ap, 0)) - r.Load(int3(bp, 0)));
    }
    return total;
}
float2 refine_one(Texture2D<float> f, Texture2D<float> r, Texture2D<float2> base, int2 p) {
    int2 bp = min(int2(p.x / 2, p.y / 2), int2(baseWidth - 1, baseHeight - 1));
    float2 prior = base.Load(int3(bp, 0)) * 2.0f;
    // Bound a bad coarse match so the wide local window still contains the
    // true correspondence. This is a GPU-only confidence guard, not a CPU
    // fallback or a scene-specific seed.
    int2 origin = clamp(int2(round(prior)), int2(-8, -8), int2(8, 8));
    float best = 1e20f; int2 v = origin;
    // Keep a generous local window after integer upsampling. This handles
    // half-sample ambiguity at the coarse level without CPU intervention.
    [loop] for (int y = -16; y <= 16; ++y) [loop] for (int x = -16; x <= 16; ++x) {
        int2 q = p + origin + int2(x, y); float e = cost(f, r, p, q);
        if (e < best) { best = e; v = q; }
    }
    // One-dimensional parabolic refinement around the integer winner. This
    // remains GPU resident and gives fractional-pixel vectors to XeSS.
    float ex0 = cost(f, r, p, v + int2(-1, 0)), ex1 = cost(f, r, p, v + int2(1, 0));
    float ey0 = cost(f, r, p, v + int2(0, -1)), ey1 = cost(f, r, p, v + int2(0, 1));
    float dx = clamp((ex0 - ex1) / max(ex0 + ex1 - 2.0f * best, 1e-5f), -0.5f, 0.5f);
    float dy = clamp((ey0 - ey1) / max(ey0 + ey1 - 2.0f * best, 1e-5f), -0.5f, 0.5f);
    // The search coordinate is absolute; the flow resource stores a
    // displacement, matching the coarse-level convention and XeSS velocity
    // input contract.
    return float2(v - p) + float2(dx, dy);
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(tid.xy); ForwardOut[p] = refine_one(FrameA, FrameB, BaseForward, p); BackwardOut[p] = refine_one(FrameB, FrameA, BaseBackward, p);
}
