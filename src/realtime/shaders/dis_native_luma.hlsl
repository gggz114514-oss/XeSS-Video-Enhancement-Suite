// Preserve original 8-bit Y bytes; do not range-expand limited Y or convert RGB.
Texture2D<float> Source : register(t0);
RWTexture2D<float> Destination : register(u0);
cbuffer Params : register(b0) { uint width; uint height; uint cropX; uint cropY; };
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    Destination[id.xy] = floor(Source.Load(int3(id.xy + uint2(cropX,cropY),0))*255.0+0.5);
}
