Texture2D<float> Current : register(t0);
Texture2D<float> Previous : register(t1);
Texture2D<float2> Motion : register(t2);
Texture2D<float> Confidence : register(t3);
Texture2D<float> Response : register(t4);
RWTexture2D<float> Output : register(u0);
cbuffer Params : register(b0) {uint w; uint h; uint iw; uint ih; uint reset; uint3 unused;};
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=w || id.y>=h) return;
    int2 p=id.xy; float d=Current[p];
    if(reset) {Output[p]=d; return;}
    float2 qf=float2(p)+Motion[p];
    bool inside=all(qf>=0) && qf.x<=w-1 && qf.y<=h-1;
    int2 q=clamp(int2(round(qf)),int2(0,0),int2(w-1,h-1));
    float lo=d,hi=d;
    for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
        float v=Current[clamp(p+int2(x,y),int2(0,0),int2(w-1,h-1))]; lo=min(lo,v);hi=max(hi,v);
    }
    float a=inside ? .25*saturate(Confidence[p])*saturate(1-Response[p])*saturate(1-(hi-lo)*10) : 0;
    Output[p]=lerp(d,Previous[q],a);
}
