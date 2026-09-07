// Fused GPU post for the full-GPU chain: fixed sharpen (0.25) + vertical
// ringing guard (0.75), a faithful port of sr_postprocess.py's fixed mode in
// the 0..255 domain, fused into one dispatch.  The XeSS output arrives as an
// RGBA8 UNORM surface (0..1) and the guide as RGBA16F (0..255, from the
// bicubic guide pass); the result is written normalized into the D3D11-owned
// shared texture oneVPL VPP imports.
Texture2D<float4> SrsFrame : register(t0);  // post-XeSS output, 0..1
Texture2D<float4> GuideFrame : register(t1); // bicubic-resized source, 0..255
RWTexture2D<float4> Output : register(u0);

cbuffer Params : register(b0) {
    uint width; uint height; uint static_strength_x100; uint guard_strength_x100;
    uint unused0; uint unused1; uint unused2; uint unused3;
};

static const float3 LUMA = float3(0.299f, 0.587f, 0.114f);

float luma3x3(int2 p, int dx, int dy) {
    return dot(SrsFrame.Load(int3(p + int2(dx, dy), 0)).rgb * 255.0f, LUMA);
}

float3 sharpen_pixel(int2 p) {
    float3 center = SrsFrame.Load(int3(p, 0)).rgb * 255.0f;
    float m[9];
    int k = 0;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
        [unroll] for (int dx = -1; dx <= 1; ++dx)
            m[k++] = luma3x3(p, dx, dy);
    float lmin = min(min(min(m[0], m[1]), min(m[2], m[3])),
                     min(min(m[4], m[5]), min(m[6], m[7])));
    float lmax = max(max(max(m[0], m[1]), max(m[2], m[3])),
                     max(max(m[4], m[5]), max(m[6], m[7])));
    float contrast = min(saturate((lmax - lmin - 72.0f) / 80.0f), 0.65f);
    const float strength = min((1.0f - contrast) *
                               static_strength_x100 / 100.0f * 1.65f, 1.65f);
    const float w[3][3] = {
        { 0.077847f, 0.123317f, 0.077847f },
        { 0.123317f, 0.195346f, 0.123317f },
        { 0.077847f, 0.123317f, 0.077847f },
    };
    float3 blur = 0.0f;
    [unroll] for (int dy2 = -1; dy2 <= 1; ++dy2)
        [unroll] for (int dx2 = -1; dx2 <= 1; ++dx2)
            blur += w[dy2 + 1][dx2 + 1] * (SrsFrame.Load(int3(p + int2(dx2, dy2), 0)).rgb * 255.0f);
    float3 detail = clamp(center - blur, -24.0f, 24.0f);
    return clamp(center + detail * strength, 0.0f, 255.0f);
}

float4 guard_pixel(int2 p) {
    float gy = dot(GuideFrame.Load(int3(p, 0)).rgb, LUMA);
    float gyL = dot(GuideFrame.Load(int3(p + int2(-1, 0), 0)).rgb, LUMA);
    float gyR = dot(GuideFrame.Load(int3(p + int2(1, 0), 0)).rgb, LUMA);
    float sobel = abs((gyR - gyL) / 2.0f) / 8.0f;
    float blend = saturate((sobel - 0.5f) / 4.0f);
    const float k5[5] = { 0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f };
    float acc = 0.0f;
    [unroll] for (int dy = -2; dy <= 2; ++dy)
    [unroll] for (int dx = -2; dx <= 2; ++dx) {
        int2 q = p + int2(dx, dy);
        float3 c = GuideFrame.Load(int3(q, 0)).rgb;
        float y = dot(c, LUMA);
        float s = abs((dot(GuideFrame.Load(int3(q + int2(1, 0), 0)).rgb, LUMA)
                       - dot(GuideFrame.Load(int3(q + int2(-1, 0), 0)).rgb, LUMA)) / 2.0f) / 8.0f;
        acc += k5[dy + 2] * k5[dx + 2] * saturate((s - 0.5f) / 4.0f);
    }
    blend = min(acc * guard_strength_x100 / 100.0f, 0.90f);
    return float4(blend, 1.0f - blend, 0, 1);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(tid.xy);
    float3 sharp = sharpen_pixel(p);
    float4 guard = guard_pixel(p);
    float3 guide = GuideFrame.Load(int3(p, 0)).rgb;
    float3 result = sharp * guard.g + guide * guard.r;
    Output[p] = float4(saturate(result / 255.0f), 1.0f);
}
