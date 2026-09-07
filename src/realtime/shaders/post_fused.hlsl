// Phase 6 GPU post: sharpen + vertical-ringing guard, fused.
// Faithful port of sr_postprocess.py's fixed-mode formulas:
//   sharpen (CAS-style with luma detail) then guard blend toward the
//   source-resized guide.  All constants copied from the CPU reference;
//   relative output must pass PSNR>=60dB / SSIM>=0.999 before defaulting.
//
// Dispatch: one thread per output pixel.  Sources: input texture (post-XeSS
// frame, rgb8), guide texture (source frame resized to output, rgb8 held as
// it is consumed by the CPU path); guards need history only for adaptive
// mode which Phase 6 defers.
Texture2D<float4> InputFrame   : register(t0);   // sharpened input (or raw)
Texture2D<float4> GuideFrame   : register(t1);   // source resized to output (cubic)
RWTexture2D<float4> Output     : register(u0);
cbuffer Params : register(b0) {
    float staticStrength;    // 0.25 (r3 fixed)
    float motionStrength;    // 0.25 (fixed ignores motion)
    float guardStrength;     // 0.75
    uint  width;
    uint  height;
};

static const float3 LUMA = float3(0.299f, 0.587f, 0.114f);

float luma3x3(Texture2D<float4> tex, int2 p, int dx, int dy) {
    return dot(tex.Load(int3(p + int2(dx, dy), 0)).rgb, LUMA);
}

// ---- sharpen (fixed mode) ------------------------------------------------
float4 sharpen_pixel(int2 p) {
    float3 center = InputFrame.Load(int3(p, 0)).rgb;

    // luma + 3x3 erode/dilate + local contrast -> noise_inv
    float m[9];
    int k = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            m[k++] = luma3x3(InputFrame, p, dx, dy);
    float lmin = min(min(min(m[0], m[1]), min(m[2], m[3])),
                     min(min(m[4], m[5]), min(m[6], m[7])));
    float lmax = max(max(max(m[0], m[1]), max(m[2], m[3])),
                     max(max(m[4], m[5]), max(m[6], m[7])));
    float contrast = saturate((lmax - lmin - 72.0f) / 80.0f);     // clip 0..0.65
    contrast = min(contrast, 0.65f);
    float noiseInv = 1.0f - contrast;
    float strength = noiseInv * staticStrength;                    // fixed mode
    strength = min(strength * 1.65f, 1.65f);

    // detail = frame - gaussian(0.8); approximated 3x3 (weights from OpenCV
    // with sigma 0.8 -> kernel radius ~1); clip detail to [-24, 24]
    float w[3][3] = {
        { 0.077847f, 0.123317f, 0.077847f },
        { 0.123317f, 0.195346f, 0.123317f },
        { 0.077847f, 0.123317f, 0.077847f },
    };
    float3 blur = 0.0f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            blur += w[dy + 1][dx + 1] * InputFrame.Load(int3(p + int2(dx, dy), 0)).rgb;
    float3 detail = clamp(center - blur, -24.0f, 24.0f);
    float3 out = center + detail * strength;
    return float4(clamp(out, 0.0f, 255.0f), 1.0f);
}

// ---- guard guide (source frame resized, sobel dx, blur) ------------------
float4 guard_pixel(int2 p) {
    float3 g = GuideFrame.Load(int3(p, 0)).rgb;
    float gy = dot(g, LUMA);
    float gyL = dot(GuideFrame.Load(int3(p + int2(-1, 0), 0)).rgb, LUMA);
    float gyR = dot(GuideFrame.Load(int3(p + int2(1, 0), 0)).rgb, LUMA);
    float sobel = abs((gyR - gyL) / 2.0f) / 8.0f;      // CPU: cv2 Sobel dx /8
    float blend = saturate((sobel - 0.5f) / 4.0f);     // clip 0..1 built-in
    // CPU GaussianBlur sigma 2.5 -> 5x5 kernel (alpha 0.4):
    const float k5[5] = { 0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f };
    // horizontal+vertical separable pass over 5x5 neighborhood of blend map
    // (implemented as two passes in a real kernel; here one-pass 5x5 sum)
    float acc = 0.0f;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++) {
            int2 q = p + int2(dx, dy);
            float3 c = GuideFrame.Load(int3(q, 0)).rgb;
            float y = dot(c, LUMA);
            float s = abs((dot(GuideFrame.Load(int3(q + int2(1, 0), 0)).rgb, LUMA)
                           - dot(GuideFrame.Load(int3(q + int2(-1, 0), 0)).rgb, LUMA)) / 2.0f) / 8.0f;
            float b = saturate((s - 0.5f) / 4.0f);
            acc += k5[dy + 2] * k5[dx + 2] * b;
        }
    blend = min(acc * guardStrength, 0.90f);
    float invBlend = 1.0f - blend;
    return float4(blend, invBlend, 0.0f, 1.0f);
}

// ---- fused output ---------------------------------------------------------
[numthreads(8, 8, 1)]
void main_cs(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(tid.xy);
    float4 sharp = sharpen_pixel(p);
    float4 guard = guard_pixel(p);
    float3 guide = GuideFrame.Load(int3(p, 0)).rgb;
    float3 out = sharp.rgb * guard.g + guide * guard.r;   // out = sharp*inv + guide*blend
    Output[p] = float4(clamp(out, 0.0f, 255.0f), 1.0f);
}
