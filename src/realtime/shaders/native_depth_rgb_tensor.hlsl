Texture2D<float4> RGB : register(t0);
StructuredBuffer<float4> Weights : register(t1);
StructuredBuffer<int> Offsets : register(t2);
StructuredBuffer<float> Normalization : register(t3);
RWByteAddressBuffer Tensor : register(u0);
RWTexture2D<float4> Resized : register(u1);
cbuffer Params:register(b0) { uint width;uint height;uint model_size;uint unused; };
[numthreads(8,8,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    if(tid.x>=model_size || tid.y>=model_size)return;
    int sx=Offsets[tid.x],sy=Offsets[model_size+tid.y];
    precise float4 ax=Weights[tid.x],ay=Weights[model_size+tid.y];
    precise float3 result=0;
    [unroll]for(int y=0;y<4;++y) {
        precise float3 row=0;
        [unroll]for(int x=0;x<4;++x) {
            int2 p=clamp(int2(sx+x-1,sy+y-1),int2(0,0),int2(width-1,height-1));
            precise float3 value=round(RGB.Load(int3(p,0)).rgb*255.0f);
            precise float3 term=value*ax[x];row=row+term;
        }
        precise float3 term=row*ay[y];result=result+term;
    }
    precise float3 bytes=clamp(round(result),0.0f,255.0f);
    Resized[tid.xy]=float4(bytes/255.0f,1);
    float3 norm=float3(Normalization[(uint)bytes.r],Normalization[256+(uint)bytes.g],Normalization[512+(uint)bytes.b]);
    uint at=tid.y*model_size+tid.x,plane=model_size*model_size;
    Tensor.Store(4*at,asuint(norm.r));Tensor.Store(4*(at+plane),asuint(norm.g));Tensor.Store(4*(at+2*plane),asuint(norm.b));
}
