// Reuse the already verified FFmpeg7.1 SSSE3 integer conversion tables.
#define main unused_uncropped_color_main
#include "surface_nv12_color.hlsl"
#undef main
cbuffer CropParams : register(b1) { uint crop_x; uint crop_y; uint spare0; uint spare1; };
[numthreads(8,8,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    if(tid.x>=width || tid.y>=height) return;
    int2 p=int2(tid.xy)+int2(crop_x,crop_y);
    int y=(int)(LumaPlane.Load(int3(p,0))*255.0f+0.5f);
    int2 uv=(int2)(ChromaPlane.Load(int3(p>>1,0))*255.0f+0.5f);
    int yc=pmulhw((y<<3)-128,9539),u=(uv.x<<3)-1024,v=(uv.y<<3)-1024;
    bool b=matrix_select!=0;
    int r=clip8(yc+pmulhw(v,b?k709Limited.vr:k601Limited.vr));
    int g=clip8(yc+pmulhw(u,b?k709Limited.ug:k601Limited.ug)+pmulhw(v,b?k709Limited.vg:k601Limited.vg));
    int blue=clip8(yc+pmulhw(u,b?k709Limited.ub:k601Limited.ub));
    ColorOut[tid.xy]=float4(float3(r,g,blue)/255.0f,1);
}
