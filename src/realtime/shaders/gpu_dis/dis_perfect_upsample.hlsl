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
    float2 p = (float2(id.xy) + 0.5) * float2(srcWidth, srcHeight) / float2(dstWidth, dstHeight) - 0.5;
    Destination[id.xy] = sample_linear(p);
}
