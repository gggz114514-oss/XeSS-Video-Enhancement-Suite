Texture2D<float> Previous : register(t0);
Texture2D<float> Current : register(t1);
Texture2D<float> Mask : register(t2);
RWStructuredBuffer<float> Output : register(u0);
cbuffer Params:register(b0) {uint w;uint h;uint2 reserved;uint4 unused;};
groupshared float diff[256];
groupshared uint reliable[256];
groupshared uint ph[32];
groupshared uint ch[32];
[numthreads(256,1,1)]
void main(uint t:SV_GroupIndex) {
    if(t<32) {ph[t]=0;ch[t]=0;}GroupMemoryBarrierWithGroupSync();
    float sum=0;uint n=0;
    for(uint i=t;i<w*h;i+=256) {
        int2 p=int2(i%w,i/w);float a=Previous[p],b=Current[p];
        sum+=abs(a-b)*255;n+=Mask[p]<.5;
        InterlockedAdd(ph[min(31,(uint)(saturate(a)*32))],1);
        InterlockedAdd(ch[min(31,(uint)(saturate(b)*32))],1);
    }
    diff[t]=sum;reliable[t]=n;GroupMemoryBarrierWithGroupSync();
    for(uint s=128;s>0;s>>=1) {if(t<s) {diff[t]+=diff[t+s];reliable[t]+=reliable[t+s];}GroupMemoryBarrierWithGroupSync();}
    if(t==0) {
        float hist=0;for(uint k=0;k<32;++k) hist+=abs((float)ph[k]-ch[k]);
        Output[0]=diff[0]/(w*h);Output[1]=hist/(2*w*h);Output[2]=(float)reliable[0]/(w*h);Output[3]=w*h;
    }
}
