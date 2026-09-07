Texture2D<float> PreviousLuma : register(t0);
Texture2D<float> CurrentLuma : register(t1);
RWTexture2D<float2> FlowOut : register(u0);
RWTexture2D<float> MaskOut : register(u1); // reserved for the shared root signature

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
};

int2 cp(int2 p) { return clamp(p, int2(0, 0), int2(int(inputWidth) - 1, int(inputHeight) - 1)); }
float patch_cost(Texture2D<float> a, Texture2D<float> b, int2 p, int2 q) {
    float cost = 0.0f;
    [unroll] for (int dy = -1; dy <= 1; ++dy) [unroll] for (int dx = -1; dx <= 1; ++dx) {
        int2 ap = cp(p + int2(dx, dy));
        int2 bp = cp(q + int2(dx, dy));
        cost += abs(a.Load(int3(ap, 0)) - b.Load(int3(bp, 0)));
    }
    return cost;
}

float2 search_one(Texture2D<float> a, Texture2D<float> b, int2 p) {
    float best = 1e20f; int2 bestDelta = 0;
    [loop] for (int dy = -8; dy <= 8; ++dy) [loop] for (int dx = -8; dx <= 8; ++dx) {
        int2 delta = int2(dx, dy);
        float c = patch_cost(a, b, p, cp(p + delta));
        if (c < best) { best = c; bestDelta = delta; }
    }
    return float2(bestDelta);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // fixed_block_v1 search now runs once per input pixel.  Keeping the
    // integer search and patch definition unchanged makes this an H1
    // organization change rather than a new optical-flow algorithm.  The
    // output-resolution velocity surface is produced by the separate light
    // upsample shader.
    if (tid.x >= inputWidth || tid.y >= inputHeight) return;
    const int2 p = int2(tid.xy);
    FlowOut[p] = search_one(CurrentLuma, PreviousLuma, p);
}
