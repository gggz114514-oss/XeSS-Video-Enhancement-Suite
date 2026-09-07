StructuredBuffer<float> Limits : register(t0);
Texture2D<float> Stable : register(t1);
Texture2D<float> Resized : register(t2);
Texture2D<float> Mask : register(t3);
RWStructuredBuffer<float> Output : register(u0);
cbuffer Params : register(b0) {uint w;uint h;uint ow;uint oh;uint4 unused;};
groupshared float4 a[256];
groupshared float4 b[256];
groupshared float4 c[256];
[numthreads(256,1,1)]
void main(uint t:SV_GroupIndex) {
    float4 sa=float4(1e30,-1e30,0,0), sb=sa, sc=float4(1e30,-1e30,0,0);
    for(uint i=t;i<w*h;i+=256) {
        int2 p=int2(i%w,i/w);float v=Stable[p],m=Mask[p];
        if(isfinite(v)) {sa.x=min(sa.x,v);sa.y=max(sa.y,v);sa.z+=v;sa.w+=v*v;} else sc.z++;
        if(isfinite(m)) {sc.x=min(sc.x,m);sc.y=max(sc.y,m);sc.w+=m;} else sc.z++;
    }
    for(uint j=t;j<ow*oh;j+=256) {
        float v=Resized[int2(j%ow,j/ow)];
        if(isfinite(v)) {sb.x=min(sb.x,v);sb.y=max(sb.y,v);sb.z+=v;sb.w+=v*v;} else sc.z++;
    }
    a[t]=sa;b[t]=sb;c[t]=sc;GroupMemoryBarrierWithGroupSync();
    for(uint s=128;s>0;s>>=1) {
        if(t<s) {
            a[t]=float4(min(a[t].x,a[t+s].x),max(a[t].y,a[t+s].y),a[t].zw+a[t+s].zw);
            b[t]=float4(min(b[t].x,b[t+s].x),max(b[t].y,b[t+s].y),b[t].zw+b[t+s].zw);
            c[t]=float4(min(c[t].x,c[t+s].x),max(c[t].y,c[t+s].y),c[t].zw+c[t+s].zw);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if(t==0) {
        for(uint k=0;k<4;++k) Output[k]=Limits[k];
        Output[4]=a[0].x;Output[5]=a[0].y;Output[6]=a[0].z/(w*h);Output[7]=a[0].w/(w*h);
        Output[8]=b[0].x;Output[9]=b[0].y;Output[10]=b[0].z/(ow*oh);Output[11]=b[0].w/(ow*oh);
        Output[12]=c[0].x;Output[13]=c[0].y;Output[14]=c[0].z;Output[15]=c[0].w/(w*h);
    }
}
