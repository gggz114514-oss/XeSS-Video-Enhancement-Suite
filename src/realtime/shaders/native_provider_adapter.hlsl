// Consumer-only adaptation. No optical-flow search is performed here.
Texture2D<float2> Forward : register(t0);
Texture2D<float2> Reverse : register(t1);
Texture2D<float> PreviousLuma : register(t2);
Texture2D<float> CurrentLuma : register(t3);
RWTexture2D<float2> HalfFlow : register(u0);
RWTexture2D<float> Reliability : register(u1);
cbuffer Params : register(b0) {uint w;uint h;uint reversePass;uint unused;uint4 padding;};
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID) {
    if(id.x>=w || id.y>=h)return;
    int2 p=int2(id.xy);
    float2 f=Forward.Load(int3(p,0));
    HalfFlow[p]=f; // Explicit R32G32->R16G16 SDK-adapter quantization.
    if(reversePass==0) {
        int2 q=p+int2(round(f));
        bool inside=all(q>=0) && q.x<int(w) && q.y<int(h);
        q=clamp(q,int2(0,0),int2(w-1,h-1));
        float2 b=Reverse.Load(int3(q,0));
        float limit=1.5+0.05*(length(f)+length(b));
        float cycle=saturate(1-length(f+b)/max(2*limit,1e-4));
        float photo=saturate(1-abs(CurrentLuma[p]-PreviousLuma[q])/0.31372549);
        Reliability[p]=inside?cycle*photo:0;
    }
}
