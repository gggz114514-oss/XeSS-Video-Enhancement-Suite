// OpenCV INTER_AREA-compatible downsample for the CV_8U DIS pyramid, exact
// semantics (Z2):
//
// - Integer-ratio paths (srcW % dstW == 0 and srcH % dstH == 0) accumulate
//   the block sum in 32-bit integer and apply the empirically verified cv2
//   rules:
//     * exact 2x (src == 2*dst): round-half-up  (sum + 2) >> 2
//     * other integer ratios (4x, 8x, ...): round-to-nearest-even on the
//       exact average (measured on cv2 5.0.0 at 4176 and 864x480 content:
//       sum%area == area/2 rounds to the even quotient; the 2x fast path
//       differs and is half-up).
// - Non-integer ratios (odd inputs): the generic cv2 resizeArea path with
//   float32 weights, sequential float accumulation and cvRound (half-away,
//   positive domain) on sum/area.
//
// The float fast-math hazards of the predecessor (midpoint truncation from
// ULPs) are gone because the integer paths never round a float.
Texture2D<float> Source : register(t0);
RWTexture2D<float> Destination : register(u0);
cbuffer Params : register(b0) { uint srcWidth; uint srcHeight; uint dstWidth; uint dstHeight; };

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dstWidth || id.y >= dstHeight) return;
    uint x = id.x, y = id.y;
    if (srcWidth % dstWidth == 0 && srcHeight % dstHeight == 0)
    {
        uint sx = srcWidth / dstWidth, sy = srcHeight / dstHeight;
        uint x0 = x * sx, y0 = y * sy;
        int sum = 0;
        for (uint yy = 0; yy < sy; ++yy)
            for (uint xx = 0; xx < sx; ++xx)
                sum += (int)Source.Load(int3(x0 + xx, y0 + yy, 0));
        uint area = sx * sy;
        int value;
        if (sx == 1 && sy == 1)
        {
            value = sum;
        }
        else if (sx == 2 && sy == 2)
        {
            // cv2 two-pixel shrink fast path: positive half-up.
            value = (sum + 2) >> 2;
        }
        else
        {
            // generic integer ratio: round-to-nearest-even of sum/area
            uint rem = (uint)sum % area;
            uint q = (uint)sum / area;
            if (rem * 2 > area) q += 1;
            else if (rem * 2 == area) q = (q & 1u) ? q + 1u : q;
            value = (int)q;
        }
        Destination[id.xy] = (float)value;
        return;
    }
    // Generic (non-integer ratio) path: exact cv2 computeResizeAreaTab taps
    // (tab geometry computed in double like cv2's computeResizeAreaTab; per-tap
    // alpha cast to float; accumulation and final v_round in float).
    double xscale = (double)srcWidth / (double)dstWidth;
    double yscale = (double)srcHeight / (double)dstHeight;
    double fsx1 = (double)x * xscale;
    double fsx2 = fsx1 + xscale;
    double cellW = min(xscale, (double)srcWidth - fsx1);
    int sx1 = (int)ceil(fsx1), sx2 = (int)floor(fsx2);
    sx2 = min(sx2, (int)srcWidth - 1);
    sx1 = min(sx1, sx2);
    // per-tap alpha table for this destination column (tab order) and taps
    const int kMaxTaps = 16;
    int tapCols[kMaxTaps];
    precise float tapAlphas[kMaxTaps];
    int nTaps = 0;
    if ((double)sx1 - fsx1 > 1e-3)
    {
        tapCols[nTaps] = max(sx1 - 1, 0);
        tapAlphas[nTaps] = (float)(((double)sx1 - fsx1) / cellW);
        nTaps++;
    }
    for (int sx = sx1; sx < sx2; ++sx)
    {
        tapCols[nTaps] = clamp(sx, 0, (int)srcWidth - 1);
        tapAlphas[nTaps] = (float)(1.0 / cellW);
        nTaps++;
    }
    if (fsx2 - (double)sx2 > 1e-3)
    {
        tapCols[nTaps] = clamp(sx2, 0, (int)srcWidth - 1);
        tapAlphas[nTaps] = (float)(min(min(fsx2 - (double)sx2, 1.0), cellW) / cellW);
        nTaps++;
    }
    double fsy1 = (double)y * yscale;
    double fsy2 = fsy1 + yscale;
    double cellH = min(yscale, (double)srcHeight - fsy1);
    int sy1 = (int)ceil(fsy1), sy2 = (int)floor(fsy2);
    sy2 = min(sy2, (int)srcHeight - 1);
    sy1 = min(sy1, sy2);
    precise float sum = 0.0;
    bool first = true;
    // y-tap rows in tab order; rowsum per source row in float32 tab order.
    if ((double)sy1 - fsy1 > 1e-3)
        {
        int sy = max(sy1 - 1, 0);
        precise float beta = (float)(((double)sy1 - fsy1) / cellH);
        precise float r = 0.0;
        for (int k = 0; k < nTaps; ++k)
            r += Source.Load(int3((uint)tapCols[k], (uint)clamp(sy, 0, (int)srcHeight - 1), 0)) * tapAlphas[k];
        sum = beta * r;
        first = false;
        }
    for (int sy = sy1; sy < sy2; ++sy)
        {
        precise float beta = (float)(1.0 / cellH);
        precise float r = 0.0;
        for (int k = 0; k < nTaps; ++k)
            r += Source.Load(int3((uint)tapCols[k], (uint)clamp(sy, 0, (int)srcHeight - 1), 0)) * tapAlphas[k];
        if (first) { sum = beta * r; first = false; }
        else sum += beta * r;
        }
    if (fsy2 - (double)sy2 > 1e-3)
        {
        int sy = clamp(sy2, 0, (int)srcHeight - 1);
        precise float beta = (float)(min(min(fsy2 - (double)sy2, 1.0), cellH) / cellH);
        precise float r = 0.0;
        for (int k = 0; k < nTaps; ++k)
            r += Source.Load(int3((uint)tapCols[k], (uint)clamp(sy, 0, (int)srcHeight - 1), 0)) * tapAlphas[k];
        if (first) sum = beta * r;
        else sum += beta * r;
        }
    // v_round: round-half-to-even, saturate to [0,255]
    float lo = floor(sum);
    float frac = sum - lo;
    float rv;
    if (frac < 0.5) rv = lo;
    else if (frac > 0.5) rv = lo + 1.0;
    else rv = (fmod(lo, 2.0) == 0.0) ? lo : lo + 1.0;
    Destination[id.xy] = clamp(rv, 0.0, 255.0);
}
