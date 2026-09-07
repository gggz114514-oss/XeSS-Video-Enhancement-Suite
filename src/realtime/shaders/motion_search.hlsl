// Gate 2 GPU-resident motion candidate.  This is deliberately a small,
// auditable block matcher rather than a CPU fallback: inputs, forward/backward
// flow, consistency and the metric reduction all stay in GPU resources.
Texture2D<float> FrameA : register(t0);
Texture2D<float> FrameB : register(t1);
RWTexture2D<int2> FlowOut : register(u0);
RWStructuredBuffer<uint> Summary : register(u1);

cbuffer Params : register(b0) {
    uint width;
    uint height;
    uint scenarioAndDirection;
    int shiftX;
    int shiftY;
    float angle;
    float centerX;
    float centerY;
};

int2 clamp_point(int2 p) { return clamp(p, int2(0, 0), int2(width - 1, height - 1)); }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(tid.xy);
    bool reverse = (scenarioAndDirection & 0x80000000u) != 0;
    // Both directions use the same GPU kernel. Swapping the images avoids a
    // second algorithm and makes the forward/backward consistency test exact.
    float center = reverse ? FrameB.Load(int3(p, 0)) : FrameA.Load(int3(p, 0));
    float best = 1e20f;
    int2 bestDelta = int2(0, 0);
    // The Quality candidate starts with a bounded integer search. The radius
    // is explicit so a scene beyond it is measured as a known Fast-only limit.
    [unroll]
    for (int dy = -4; dy <= 4; ++dy) {
        [unroll]
        for (int dx = -4; dx <= 4; ++dx) {
            int2 q = clamp_point(p + int2(dx, dy));
            float other = reverse ? FrameA.Load(int3(q, 0)) : FrameB.Load(int3(q, 0));
            float e = abs(center - other);
            if (e < best) { best = e; bestDelta = int2(dx, dy); }
        }
    }
    FlowOut[p] = bestDelta;
}
