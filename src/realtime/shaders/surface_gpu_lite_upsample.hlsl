// GPU Block Lite: restore a reduced-grid fixed_block_v1 flow/confidence field
// to the full input grid consumed by the existing mask and velocity passes.
// Flow components are restored independently for non-square scale factors.
Texture2D<float2> FlowIn : register(t0);
Texture2D<float> ConfidenceIn : register(t1);
RWTexture2D<float2> FlowOut : register(u0);
RWTexture2D<float> ConfidenceOut : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
    uint liteWidth; uint liteHeight; uint unused2; uint unused3;
};

int2 cp_lite(int2 p) {
    return clamp(p, int2(0, 0), int2(int(liteWidth) - 1, int(liteHeight) - 1));
}

float2 sample_flow(float2 source) {
    const int2 base = int2(floor(source));
    const float2 f = frac(source);
    const float2 a = FlowIn.Load(int3(cp_lite(base + int2(0, 0)), 0));
    const float2 b = FlowIn.Load(int3(cp_lite(base + int2(1, 0)), 0));
    const float2 c = FlowIn.Load(int3(cp_lite(base + int2(0, 1)), 0));
    const float2 d = FlowIn.Load(int3(cp_lite(base + int2(1, 1)), 0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

float sample_confidence(float2 source) {
    const int2 base = int2(floor(source));
    const float2 f = frac(source);
    const float a = ConfidenceIn.Load(int3(cp_lite(base + int2(0, 0)), 0));
    const float b = ConfidenceIn.Load(int3(cp_lite(base + int2(1, 0)), 0));
    const float c = ConfidenceIn.Load(int3(cp_lite(base + int2(0, 1)), 0));
    const float d = ConfidenceIn.Load(int3(cp_lite(base + int2(1, 1)), 0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    const float2 source = (float2(tid.xy) + 0.5f) *
                          float2(liteWidth / float(width),
                                 liteHeight / float(height)) - 0.5f;
    const float2 lite_flow = sample_flow(source);
    FlowOut[tid.xy] = lite_flow *
                       float2(inputWidth / float(liteWidth),
                              inputHeight / float(liteHeight));
    ConfidenceOut[tid.xy] = sample_confidence(source);
}
