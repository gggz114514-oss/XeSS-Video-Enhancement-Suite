// Coarse-to-fine DIS flow initialization.  OpenCV resizes each component
// linearly and multiplies by two when moving to the next finer pyramid level.
Texture2D<float2> Source : register(t0);
RWTexture2D<float2> Destination : register(u0);
cbuffer Params : register(b0) { uint srcWidth; uint srcHeight; uint dstWidth; uint dstHeight; };

float2 sample_linear(float2 p)
{
    p.x = clamp(p.x, 0.0, (float)srcWidth - 1.0);
    p.y = clamp(p.y, 0.0, (float)srcHeight - 1.0);
    uint2 lo = uint2(floor(p));
    uint2 hi = min(lo + 1, uint2(srcWidth - 1, srcHeight - 1));
    float2 f = p - (float2)lo;
    float2 a = Source.Load(int3(lo, 0));
    float2 b = Source.Load(int3(uint2(hi.x, lo.y), 0));
    float2 c = Source.Load(int3(uint2(lo.x, hi.y), 0));
    float2 d = Source.Load(int3(hi, 0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y) * 2.0;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dstWidth || id.y >= dstHeight) return;
#ifdef DIS_IPP_UPSAMPLE
    // Frozen OpenCV wheel's IPP single-component INTER_LINEAR path. Keep
    // coordinate construction in binary64 until the fractional weight cast,
    // and match the three explicit FP32 fused lerps (see C oracle evidence).
    precise double px = ((double)id.x + 0.5) * (double)srcWidth / (double)dstWidth - 0.5;
    precise double py = ((double)id.y + 0.5) * (double)srcHeight / (double)dstHeight - 0.5;
    int ix = (int)px, iy = (int)py;
    if (px < (double)ix) --ix;
    if (py < (double)iy) --iy;
    precise float fx = (float)(px - (double)ix), fy = (float)(py - (double)iy);
    int x0 = clamp(ix, 0, (int)srcWidth - 1), x1 = clamp(ix + 1, 0, (int)srcWidth - 1);
    int y0 = clamp(iy, 0, (int)srcHeight - 1), y1 = clamp(iy + 1, 0, (int)srcHeight - 1);
    float2 a = Source.Load(int3(x0,y0,0)), b = Source.Load(int3(x1,y0,0));
    float2 c = Source.Load(int3(x0,y1,0)), d = Source.Load(int3(x1,y1,0));
    precise float2 ab = b - a, cd = d - c;
    precise float2 top = mad(ab, fx, a), bottom = mad(cd, fx, c);
    precise float2 span = bottom - top;
    precise float2 value = mad(span, fy, top);
    Destination[id.xy] = value * 2.0;
#else
    float2 p = (float2(id.xy) + 0.5) * float2(srcWidth, srcHeight) / float2(dstWidth, dstHeight) - 0.5;
    Destination[id.xy] = sample_linear(p);
#endif
}
