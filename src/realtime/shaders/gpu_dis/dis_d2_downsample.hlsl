// Gate D2 diagnostic only. For the even dimensions used by the first D2
// closure, this is exactly the 2x2 area average behind OpenCV INTER_AREA.
Texture2D<float> Source : register(t0);
RWTexture2D<float> Destination : register(u0);
cbuffer Params : register(b0) { uint srcWidth; uint srcHeight; uint dstWidth; uint dstHeight; };

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dstWidth || id.y >= dstHeight) return;
    uint2 p = id.xy * 2;
    uint2 q = uint2(min(p.x + 1, srcWidth - 1), min(p.y + 1, srcHeight - 1));
    float area = 0.25 * (
        Source.Load(int3(p.x, p.y, 0)) +
        Source.Load(int3(q.x, p.y, 0)) +
        Source.Load(int3(p.x, q.y, 0)) +
        Source.Load(int3(q.x, q.y, 0)));
    // OpenCV's CV_8U INTER_AREA path rounds positive 2x2 averages half up.
    // Keep that quantization because every DIS pyramid level is CV_8U.
    Destination[id.xy] = floor(area + 0.5);
}
