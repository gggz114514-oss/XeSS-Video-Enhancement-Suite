StructuredBuffer<float> Input : register(t0);
StructuredBuffer<float> Limits : register(t1);
RWTexture2D<float> Output : register(u0);
cbuffer Params : register(b0) {uint w; uint h; uint iw; uint ih; uint4 unused;};
float cubic(float x) {x=abs(x); return x<=1 ? 1.5*x*x*x-2.5*x*x+1 : x<2 ? -.5*x*x*x+2.5*x*x-4*x+2 : 0;}
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=w || id.y>=h) return;
    float2 p=(float2(id.xy)+.5)*float2(iw,ih)/float2(w,h)-.5;
    int2 b=int2(floor(p)); float v=0, total=0;
    for(int y=-1;y<=2;++y) for(int x=-1;x<=2;++x) {
        int2 q=clamp(b+int2(x,y),int2(0,0),int2(iw-1,ih-1));
        float s=Input[q.y*iw+q.x];
        float a=cubic(p.x-b.x-x)*cubic(p.y-b.y-y);
        if(isfinite(s)) {v+=s*a; total+=a;}
    }
    float range=Limits[1]-Limits[0];
    Output[id.xy]=(range>1e-6 && total>1e-6) ? saturate((v/total-Limits[0])/range) : .5;
}
