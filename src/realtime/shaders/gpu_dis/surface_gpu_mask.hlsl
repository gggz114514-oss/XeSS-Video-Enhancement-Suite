// Responsive-pixel mask approximation for the GPU-resident SR path.
//
// XeSS expects an input-resolution R8 mask whose values are clipped by
// xessSetMaxResponsiveMaskValue.  The formal CPU reference combines flow
// consistency, photometric residual, outside-image coverage and (when
// present) depth edges.  This pass implements the first three terms without
// leaving the GPU: bidirectional patch consistency plus a BT.709 luma
// residual.  Depth-aware dilation remains a declared Gate 5 item.
Texture2D<float> PreviousLuma : register(t0);
Texture2D<float> CurrentLuma : register(t1);
RWTexture2D<float2> FlowOut : register(u0); // unused, kept for shared root
RWTexture2D<float> MaskOut : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
};

int2 cp(int2 p) {
    return clamp(p, int2(0, 0), int2(int(inputWidth) - 1, int(inputHeight) - 1));
}

float patch_cost(Texture2D<float> a, Texture2D<float> b, int2 p, int2 q) {
    float cost = 0.0f;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
        [unroll] for (int dx = -1; dx <= 1; ++dx) {
            int2 ap = cp(p + int2(dx, dy));
            int2 bp = cp(q + int2(dx, dy));
            cost += abs(a.Load(int3(ap, 0)) - b.Load(int3(bp, 0)));
        }
    return cost;
}

float2 search_one(Texture2D<float> a, Texture2D<float> b, int2 p) {
    float best = 1e20f;
    int2 bestDelta = 0;
    [loop] for (int dy = -8; dy <= 8; ++dy)
        [loop] for (int dx = -8; dx <= 8; ++dx) {
            int2 delta = int2(dx, dy);
            float cost = patch_cost(a, b, p, cp(p + delta));
            if (cost < best) { best = cost; bestDelta = delta; }
        }
    return float2(bestDelta);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= inputWidth || tid.y >= inputHeight) return;
    int2 p = int2(tid.xy);
    float2 f = search_one(CurrentLuma, PreviousLuma, p);
    int2 q = cp(p + int2(round(f)));
    float2 b = search_one(PreviousLuma, CurrentLuma, q);
    float consistency = length(f + b);
    float magnitude = length(f) + length(b);
    float limit = 1.5f + 0.05f * magnitude;
    float confidence = saturate(1.0f - consistency / max(2.0f * limit, 1e-4f));
    float residual = abs(CurrentLuma.Load(int3(p, 0)) -
                         PreviousLuma.Load(int3(q, 0)));
    // Luma is normalized [0,1]; the CPU reference divides 8-bit residual by
    // 80, hence 1/80 in 8-bit units is approximately 1/0.3137 here.
    float luma_response = saturate(residual / 0.31372549f) * 0.65f;
    bool inside = all((p + f) >= 0) && p.x + f.x <= inputWidth - 1 &&
                  p.y + f.y <= inputHeight - 1;
    float response = max(1.0f - confidence, luma_response);
    response = max(response, inside ? 0.0f : 1.0f);
    // Matches the formal default --responsive-max 0.8 contract.
    MaskOut[p] = min(response, 0.8f);
}
