// Phase 3 GPU post, step 1: fixed sharpen only (vertical-ringing guard and
// adaptive follow in later steps).
//
// Faithful port of sr_postprocess.py fixed-mode sharpen_frame, operating in
// the CPU's 0..255 domain (Texture2D<float4>.Load returns 0..1, so every
// value is scaled by 255 before the CPU constants 72/80/24 apply):
//   gray    = OpenCV integer fixed-point luma (9798,19235,3735,>>15)
//   erode/dilate borders = 0      (cv2 morphology default border value)
//   contrast= (lmax-lmin-72)/80   clip 0..0.65
//   strength= (1-contrast)*static*1.65
//   blur    = gaussian 7x7 sigma 0.8 with cv2 BORDER_REFLECT_101
//   detail  = frame - blur        clip +-24
//   out     = frame + detail*strength, clamp 0..255, TRUNCATE to u8
// The UNORM store rounds, so truncation is done explicitly with floor()
// (in the 0..255 domain) before dividing back to 0..1.
//
// Dispatch: one thread per output pixel, 8x8 groups.  Sits directly after
// the XeSS output texture and before readback (no GPU->CPU->GPU).
Texture2D<float4> InputFrame   : register(t0);   // post-XeSS frame (0..1)
RWTexture2D<float4> Output     : register(u0);
cbuffer Params : register(b0) {
    uint  width;
    uint  height;
    float staticStrength;    // r3 fixed = 0.25
    float pad0;
};

// cv2 BORDER_REFLECT_101 (len >= 2): "-1 -> 1, -2 -> 2, width -> width-2"
// Derived from copyMakeBorder: sequence "...4 3 2 1|0 1 2 3 4|3 2 1 0..."
int refl101(int v, int lim) {
    int r = v < 0 ? -v : v;                    // abs
    r = r % (2 * lim - 2);
    if (r > lim - 1)
        r = 2 * lim - 2 - r;
    return r;
}

float3 rgb255(int2 p) {
    // UNORM stores k/255: multiplying back can land epsilon below k, which
    // floor() then truncates to k-1.  Recover the exact CPU integer with
    // round-half-up (epsilon is ~1e-7, far from the 0.5 decision point).
    return floor(InputFrame.Load(int3(p, 0)).rgb * 255.0f + 0.5f);
}

// u8 luma with cv2 cvtColor(RGB2GRAY) quantization.  OpenCV's uint8 path is
// integer 15-bit fixed point, EXACTLY  (9798*R + 19235*G + 3735*B + 16384)
// >> 15 -- verified against all 256^3 RGB combinations (0 mismatches);
// a float dot (0.299/0.587/0.114) diverges on 0.127% of pixels by +-1 LSB,
// which is where the old min-PSNR ~58..60 dB band came from.
// border: cv2 erode's outside value is a neutral HIGH (255, edges are not
// pulled down); dilate's outside value is a neutral LOW (0).
float gray_u8(int2 p, int dx, int dy, float border) {
    int2 q = p + int2(dx, dy);
    if (q.x < 0 || q.x >= width || q.y < 0 || q.y >= height)
        return border;
    float3 rgb = InputFrame.Load(int3(q, 0)).rgb;
    uint r = (uint)floor(rgb.r * 255.0f + 0.5f);
    uint g = (uint)floor(rgb.g * 255.0f + 0.5f);
    uint b = (uint)floor(rgb.b * 255.0f + 0.5f);
    return (float)((9798u * r + 19235u * g + 3735u * b + 16384u) >> 15);
}

// gaussian sigma 0.8 -> cv2 ksize 7x7 (getGaussianKernel(7, 0.8, CV_64F)
// outer product, float32 literals).  The blur accumulation runs in double:
// a 49-term float32 sum carries +-~4e-7 of rounding noise, which lands
// floor(clamp(outC)) one LSB below the pixel value on ~9% of pixels on flat
// dark content (the stable PSNR ~58.4 dB band).  Integer-exact accumulation
// makes flat input come out exactly v (matching the CPU reference, whose own
// OpenCV FMA filter noise stays below the f32 half-ULP at the final add).
static const float W7[7][7] = {
    { 0.000000194254715f, 0.00000965682564f, 0.000100626434f,
      0.000219788338f, 0.000100626434f, 0.00000965682564f, 0.000000194254715f },
    { 0.00000965682564f, 0.000480061867f, 0.00500235953f,
      0.0109261577f, 0.00500235953f, 0.000480061867f, 0.00000965682564f },
    { 0.000100626434f, 0.00500235953f, 0.0521257832f,
      0.113853178f, 0.0521257832f, 0.00500235953f, 0.000100626434f },
    { 0.000219788338f, 0.0109261577f, 0.113853178f,
      0.248678204f, 0.113853178f, 0.0109261577f, 0.000219788338f },
    { 0.000100626434f, 0.00500235953f, 0.0521257832f,
      0.113853178f, 0.0521257832f, 0.00500235953f, 0.000100626434f },
    { 0.00000965682564f, 0.000480061867f, 0.00500235953f,
      0.0109261577f, 0.00500235953f, 0.000480061867f, 0.00000965682564f },
    { 0.000000194254715f, 0.00000965682564f, 0.000100626434f,
      0.000219788338f, 0.000100626434f, 0.00000965682564f, 0.000000194254715f },
};

// Per-channel double accumulation: a 49-term float32 sum carries +-~4e-7 of
// rounding noise, which lands floor(clamp(outC)) one LSB below the pixel
// value on ~9% of pixels on flat dark content (the stable PSNR ~58.4 dB
// band).  Exact accumulation makes flat input come out exactly v, matching
// the CPU reference (whose own OpenCV filter noise stays below the f32
// half-ULP at the final add).
float3 blur7(int2 p) {
    double3 acc = 0.0;
    for (int iy = -3; iy <= 3; iy++) {
        int y = refl101(p.y + iy, height);
        for (int ix = -3; ix <= 3; ix++) {
            int x = refl101(p.x + ix, width);
            acc += (double)W7[iy + 3][ix + 3] * (double3)rgb255(int2(x, y));
        }
    }
    return (float3)acc;
}

float4 sharpen_pixel(int2 p) {
    float3 center = rgb255(p);

    // luma + 3x3 erode/dilate (borders contribute 0, like cv2) -> contrast
    float mMin[9];
    float mMax[9];
    int k = 0;
    for (int iy = -1; iy <= 1; iy++)
        for (int ix = -1; ix <= 1; ix++) {
            mMin[k] = gray_u8(p, ix, iy, 255.0f);   // erode: neutral high
            mMax[k] = gray_u8(p, ix, iy, 0.0f);     // dilate: neutral low
            k++;
        }
    float lmin = mMin[0];
    float lmax = mMax[0];
    for (int i = 1; i < 9; i++) {
        lmin = min(lmin, mMin[i]);
        lmax = max(lmax, mMax[i]);
    }
    float contrast = saturate((lmax - lmin - 72.0f) / 80.0f);
    contrast = min(contrast, 0.65f);
    float noiseInv = 1.0f - contrast;
    float strength = noiseInv * staticStrength;     // fixed mode
    strength *= 1.65f;

    float3 detail = clamp(center - blur7(p), -24.0f, 24.0f);
    float3 outC = center + detail * strength;
    // truncation (CPU numpy unsafe copyto) via floor in the 0..255 domain
    float3 q = floor(clamp(outC, 0.0f, 255.0f));
    return float4(q / 255.0f, 1.0f);
}

[numthreads(8, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height)
        return;
    Output[tid.xy] = sharpen_pixel(int2(tid.x, tid.y));
}
