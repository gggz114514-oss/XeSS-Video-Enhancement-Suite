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
Texture2D<float2> ForwardFlow : register(t2);
Texture2D<float2> BackwardFlow : register(t3);
// Forward-search confidence published by surface_gpu_motion_tile.  Only
// consumed when constants[4] (repair mode) is non-zero, so the mode-0
// ablation keeps the original mask semantics bit-for-bit.
Texture2D<float> ForwardConfidence : register(t4);
RWTexture2D<float> ReservedOut : register(u0); // shared root signature slot
RWTexture2D<float> MaskOut : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
    uint repairMode; uint unused1; uint unused2; uint unused3;
};

int2 cp(int2 p) {
    return clamp(p, int2(0, 0), int2(int(inputWidth) - 1, int(inputHeight) - 1));
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= inputWidth || tid.y >= inputHeight) return;
    int2 p = int2(tid.xy);
    // H1: both searches are performed once by surface_gpu_motion.  The mask
    // pass only consumes the saved fields, eliminating the previous second
    // forward+backward candidate search per pixel.
    float2 f = ForwardFlow.Load(int3(p, 0));
    int2 q = cp(p + int2(round(f)));
    float2 b = BackwardFlow.Load(int3(q, 0));
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
    // Repair modes additionally mark search ties: a low best/second-best
    // confidence means the block winner is arbitrary even when the round
    // trip happens to agree.  Half weight keeps the repair from flooding
    // the responsive mask on flat regions.
    if (repairMode != 0u) {
        float search_confidence = ForwardConfidence.Load(int3(p, 0));
        response = max(response, (1.0f - search_confidence) * 0.4f);
    }
    // Matches the formal default --responsive-max 0.8 contract.
    MaskOut[p] = min(response, 0.8f);
}
