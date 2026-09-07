RWTexture2D<float2> DFlow : register(u0);
cbuffer Params : register(b0) { uint width; uint height; };
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) { if (id.x < width && id.y < height) DFlow[id.xy] = 0.0.xx; }
