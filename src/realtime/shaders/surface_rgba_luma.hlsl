// Convert the RGB application back buffer into the normalized luma surface
// consumed by the GPU Block Motion search.  The integer coefficients match
// OpenCV RGB2GRAY for 8-bit RGB, so the search sees deterministic input even
// though the source texture is sampled as UNORM.
Texture2D<float4> CurrentColor : register(t0);
Texture2D<float4> Reserved1 : register(t1);
Texture2D<float4> Reserved2 : register(t2);
Texture2D<float4> Reserved3 : register(t3);
Texture2D<float4> Reserved4 : register(t4);
RWTexture2D<float> LumaOut : register(u0);
RWTexture2D<float> ReservedOut : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint unused0; uint unused1;
    uint unused2; uint unused3;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    const float3 sampled = saturate(CurrentColor.Load(int3(tid.xy, 0)).rgb);
    const uint3 rgb = uint3(floor(sampled * 255.0f + 0.5f));
    const uint gray = (9798u * rgb.r + 19235u * rgb.g + 3735u * rgb.b +
                       16384u) >> 15;
    LumaOut[tid.xy] = gray * (1.0f / 255.0f);
}
