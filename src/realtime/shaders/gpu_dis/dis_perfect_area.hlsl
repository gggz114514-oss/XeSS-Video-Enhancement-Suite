// OpenCV INTER_AREA-compatible downsample for the CV_8U DIS pyramid.  The
// source/destination dimensions are arbitrary (including odd sizes).  The
// bounded overlap loops are the area kernel; the result is quantized back to
// the positive CV_8U domain before being stored in R32F.
Texture2D<float> Source : register(t0);
RWTexture2D<float> Destination : register(u0);
cbuffer Params : register(b0) { uint srcWidth; uint srcHeight; uint dstWidth; uint dstHeight; };

float round_even(float value)
{
    float lo = floor(value), frac = value - lo;
    if (frac < 0.5) return lo;
    if (frac > 0.5) return lo + 1.0;
    return (fmod(lo, 2.0) == 0.0) ? lo : lo + 1.0;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dstWidth || id.y >= dstHeight) return;
    float x0 = (float)id.x * (float)srcWidth / (float)dstWidth;
    float x1 = (float)(id.x + 1) * (float)srcWidth / (float)dstWidth;
    float y0 = (float)id.y * (float)srcHeight / (float)dstHeight;
    float y1 = (float)(id.y + 1) * (float)srcHeight / (float)dstHeight;
    int ix0 = (int)floor(x0), ix1 = (int)ceil(x1) - 1;
    int iy0 = (int)floor(y0), iy1 = (int)ceil(y1) - 1;
    float sum = 0.0;
    for (int y = iy0; y <= iy1; ++y) {
        float wy = max(0.0, min(y1, (float)y + 1.0) - max(y0, (float)y));
        for (int x = ix0; x <= ix1; ++x) {
            float wx = max(0.0, min(x1, (float)x + 1.0) - max(x0, (float)x));
            uint sx = (uint)clamp(x, 0, (int)srcWidth - 1);
            uint sy = (uint)clamp(y, 0, (int)srcHeight - 1);
            sum += wx * wy * Source.Load(int3(sx, sy, 0));
        }
    }
    float area = max((x1 - x0) * (y1 - y0), 1e-20);
    float value = sum / area;
    // OpenCV's two-pixel shrink path uses the integer 2x2 area fast path
    // (positive half-up); the direct 4x/odd area path uses cvRound's
    // nearest-even behavior. Keep both paths explicit for pyramid parity.
    Destination[id.xy] = (srcWidth == dstWidth * 2 && srcHeight == dstHeight * 2)
        ? floor(value + 0.5) : round_even(value);
}
