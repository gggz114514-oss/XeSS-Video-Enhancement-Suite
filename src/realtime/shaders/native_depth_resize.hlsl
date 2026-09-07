Texture2D<float> Input : register(t0);
RWTexture2D<float> Output : register(u0);
cbuffer Params : register(b0) {uint w; uint h; uint iw; uint ih; uint4 unused;};
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=w || id.y>=h) return;
    float2 p=(float2(id.xy)+.5)*float2(iw,ih)/float2(w,h)-.5; int2 b=int2(floor(p)); float2 f=frac(p);
    int2 hi=int2(iw-1,ih-1);
    float a=Input.Load(int3(clamp(b,int2(0,0),hi),0));
    float c=Input.Load(int3(clamp(b+int2(1,0),int2(0,0),hi),0));
    float d=Input.Load(int3(clamp(b+int2(0,1),int2(0,0),hi),0));
    float e=Input.Load(int3(clamp(b+int2(1,1),int2(0,0),hi),0));
    Output[id.xy]=lerp(lerp(a,c,f.x),lerp(d,e,f.x),f.y);
}
