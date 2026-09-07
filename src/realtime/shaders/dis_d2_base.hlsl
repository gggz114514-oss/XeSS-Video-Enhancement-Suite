// Gate D2 diagnostic only: convert an R8_UNORM luma surface to the R32F
// intensity representation used by the first correctness path.
Texture2D<float> Source : register(t0);
RWTexture2D<float> Destination : register(u0);
cbuffer Params : register(b0) { uint srcWidth; uint srcHeight; uint dstWidth; uint dstHeight; };

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dstWidth || id.y >= dstHeight) return;
    // Preserve the CV_8U value exactly when expanding the normalized R8
    // surface.  The CPU DIS pyramid is quantized at every INTER_AREA level.
    Destination[id.xy] = floor(Source.Load(int3(min(id.x, srcWidth - 1), min(id.y, srcHeight - 1), 0)) * 255.0 + 0.5);
}
