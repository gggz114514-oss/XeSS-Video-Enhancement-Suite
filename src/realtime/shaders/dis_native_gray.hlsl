// Motion-input adapter, not a decoder. Reuses A's surface_nv12_color.hlsl
// FFmpeg 7.1 SSSE3 integer organization, followed by OpenCV RGB2GRAY.
// Only 8-bit limited BT.601/709 is accepted by the host. No full-range guess.
Texture2D<float> Luma : register(t0);
Texture2D<float2> Chroma : register(t1);
RWTexture2D<float> Gray : register(u0);
cbuffer Params : register(b0) {
    uint width; uint height; uint cropX; uint cropY; uint matrix709; uint reserved;
};
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width || id.y>=height) return;
    uint2 p=id.xy+uint2(cropX,cropY);
    int y=(int)(Luma.Load(int3(p,0))*255.0+0.5);
    int2 uv=(int2)(Chroma.Load(int3(p>>1,0))*255.0+0.5);
    int yc=(((y<<3)-128)*9539)>>16;
    int u=(uv.x<<3)-1024,v=(uv.y<<3)-1024;
    uint r=(uint)clamp(yc+((v*(matrix709?14686:13075))>>16),0,255);
    uint g=(uint)clamp(yc+((u*(matrix709?-1747:-3209))>>16)+((v*(matrix709?-4366:-6660))>>16),0,255);
    uint b=(uint)clamp(yc+((u*(matrix709?17305:16525))>>16),0,255);
    Gray[id.xy]=(float)((9798u*r+19235u*g+3735u*b+16384u)>>15);
}
