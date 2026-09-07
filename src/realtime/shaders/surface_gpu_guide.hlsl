// Guide pass for the fused GPU post: bicubic upscale of the input-resolution
// color surface to the output resolution, emitted in the 0..255 domain the
// CPU sr_postprocess guard operates on.  OpenCV INTER_CUBIC (a = -0.75)
// weight function; the guide only drives the guard blend map, so small
// coefficient differences versus the CPU reference are measured, not assumed
// away.
Texture2D<float4> ColorIn : register(t0);   // input resolution, 0..1
RWTexture2D<float4> GuideOut : register(u0); // output resolution, 0..255 RGBA16F

cbuffer Params : register(b0) {
    uint out_width; uint out_height; uint in_width; uint in_height;
    uint unused0; uint unused1; uint unused2; uint unused3;
}

float cubic_weight(float x) {
    const float a = -0.75f;
    x = abs(x);
    if (x < 1.0f)
        return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f)
        return a * (x * (x * (x - 5.0f) + 8.0f) - 4.0f);
    return 0.0f;
}

float4 sample_bicubic(float2 src) {
    const float2 base = floor(src - 0.5f) + 0.5f;
    const float2 frac = src - base;
    float4 result = 0.0f;
    [unroll] for (int dy = -1; dy <= 2; ++dy)
    [unroll] for (int dx = -1; dx <= 2; ++dx) {
        const float wx = cubic_weight((dx - frac.x));
        const float wy = cubic_weight((dy - frac.y));
        const int2 p = int2(base + float2(dx, dy));
        const int2 q = clamp(p, int2(0, 0), int2(int(in_width) - 1, int(in_height) - 1));
        result += wx * wy * ColorIn.Load(int3(q, 0));
    }
    return result;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= out_width || tid.y >= out_height) return;
    const float2 src = (tid.xy + 0.5f) * float2(in_width / float(out_width),
                                                in_height / float(out_height)) - 0.5f;
    GuideOut[tid.xy] = float4(saturate(sample_bicubic(src)).rgb * 255.0f, 1.0f);
}
