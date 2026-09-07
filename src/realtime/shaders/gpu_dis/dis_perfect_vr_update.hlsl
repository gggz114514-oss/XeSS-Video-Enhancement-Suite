Texture2D<float2> BaseFlow : register(t0);
Texture2D<float2> DFlow : register(t1);
RWTexture2D<float2> WorkFlow : register(u0);
cbuffer Params : register(b0) { uint width; uint height; };
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    int2 p = int2(id.xy); WorkFlow[p] = BaseFlow.Load(int3(p,0)) + DFlow.Load(int3(p,0));
}
