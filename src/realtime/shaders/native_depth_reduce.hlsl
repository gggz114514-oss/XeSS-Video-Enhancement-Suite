StructuredBuffer<float> Input : register(t0);
RWStructuredBuffer<float> Limits : register(u0);
cbuffer Params : register(b0) { uint w; uint h; uint iw; uint ih; uint4 unused; };
groupshared float lows[256];
groupshared float highs[256];
groupshared uint counts[256];
[numthreads(256,1,1)]
void main(uint t : SV_GroupIndex) {
    float lo = 3.402823466e38f, hi = -3.402823466e38f; uint n = 0;
    for (uint i=t; i<iw*ih; i+=256) {
        float v=Input[i]; if (isfinite(v)) {lo=min(lo,v); hi=max(hi,v); ++n;}
    }
    lows[t]=lo; highs[t]=hi; counts[t]=n; GroupMemoryBarrierWithGroupSync();
    for(uint step=128; step>0; step>>=1) {
        if(t<step) {lows[t]=min(lows[t],lows[t+step]); highs[t]=max(highs[t],highs[t+step]); counts[t]+=counts[t+step];}
        GroupMemoryBarrierWithGroupSync();
    }
    if(t==0) {Limits[0]=lows[0]; Limits[1]=highs[0]; Limits[2]=counts[0]; Limits[3]=iw*ih-counts[0];}
}
