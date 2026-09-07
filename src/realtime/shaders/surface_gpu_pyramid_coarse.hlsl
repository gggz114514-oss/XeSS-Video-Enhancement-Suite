// H3 Fast pyramid coarse search.  At quarter resolution radius 8 covers
// approximately +/-32 input pixels.  Fine levels do not repeat this window.
Texture2D<float> PreviousQuarter : register(t0);
Texture2D<float> CurrentQuarter : register(t1);
Texture2D<float> Reserved0 : register(t2);
Texture2D<float> Reserved1 : register(t3);
RWTexture2D<float2> ForwardQuarter : register(u0);
RWTexture2D<float2> BackwardQuarter : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
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

float2 search(Texture2D<float> a, Texture2D<float> b, int2 p) {
    float best = 1e20f;
    int2 best_delta = 0;
    int best_norm = 0x7fffffff;
    [loop] for (int dy = -8; dy <= 8; ++dy)
        [loop] for (int dx = -8; dx <= 8; ++dx) {
            const int2 delta = int2(dx, dy);
            const float cost = patch_cost(a, b, p, cp(p + delta));
            const int norm = dx * dx + dy * dy;
            if (cost < best || (cost == best && norm < best_norm)) {
                best = cost;
                best_delta = delta;
                best_norm = norm;
            }
        }
    return float2(best_delta);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    const int2 p = int2(tid.xy);
    ForwardQuarter[p] = search(CurrentQuarter, PreviousQuarter, p);
    BackwardQuarter[p] = search(PreviousQuarter, CurrentQuarter, p);
}
