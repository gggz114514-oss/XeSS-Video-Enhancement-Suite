// Optional SR effects, before FG. Pixel data never leaves D3D12.
// Guard: source cubic guide -> abs(Sobel-x)/8 -> threshold -> Gaussian(2.5)
// -> strength .90, matching the accepted CPU guide algorithm (not old k5).
// Fusion: causal current + four previous sources; reuse current->previous
// motion, reject out-of-bounds/photometric mismatches and reset at scene cuts.
Texture2D<float4> Sr : register(t0);
Texture2D<float4> Source : register(t1);
Texture2D<float4> Guide : register(t2);
Texture2D<float> Mask : register(t3);
Texture2D<float> Horizontal : register(t4);
Texture2D<float4> History[4] : register(t5);
Texture2D<float2> Flow[4] : register(t9);
RWTexture2D<float4> ColorOut : register(u0);
RWTexture2D<float> MaskOut : register(u1);
cbuffer Params : register(b0) {
    uint w; uint h; uint iw; uint ih;
    uint stage; uint guardOn; uint fusionOn; uint historyCount;
};
int reflect101(int p,int n) {
    if(n<=1)return 0;
    int period=2*(n-1); p=((p%period)+period)%period;
    return p<n?p:period-p;
}
int2 reflectp(int2 p) {return int2(reflect101(p.x,w),reflect101(p.y,h));}
float cubic(float x) {
    x=abs(x); const float a=-.75;
    return x<=1 ? ((a+2)*x-(a+3))*x*x+1 : x<2 ? ((a*x-5*a)*x+8*a)*x-4*a : 0;
}
float3 cubicGuide(int2 p) {
    float2 q=(float2(p)+.5)*float2(iw,ih)/float2(w,h)-.5;
    int2 b=int2(floor(q)); float2 f=q-b; float3 sum=0;
    for(int y=-1;y<=2;++y) for(int x=-1;x<=2;++x)
        sum+=Source.Load(int3(clamp(b+int2(x,y),int2(0,0),int2(iw-1,ih-1)),0)).rgb
            *cubic(x-f.x)*cubic(y-f.y);
    return floor(saturate(sum)*255+.5)/255;
}
float gray(int2 p) {return dot(Guide.Load(int3(reflectp(p),0)).rgb*255,float3(.299,.587,.114));}
float gaussian(int i) {return exp(-float(i*i)/12.5);}
float3 bilinearColor(Texture2D<float4> tex,float2 p) {
    p=clamp(p,0,float2(iw-1,ih-1)); int2 b=int2(floor(p)),e=min(b+1,int2(iw-1,ih-1));float2 f=p-b;
    return lerp(lerp(tex.Load(int3(b,0)).rgb,tex.Load(int3(e.x,b.y,0)).rgb,f.x),
                lerp(tex.Load(int3(b.x,e.y,0)).rgb,tex.Load(int3(e,0)).rgb,f.x),f.y)*255;
}
float2 bilinearFlow(Texture2D<float2> tex,float2 p) {
    p=clamp(p,0,float2(iw-1,ih-1)); int2 b=int2(floor(p)),e=min(b+1,int2(iw-1,ih-1));float2 f=p-b;
    return lerp(lerp(tex.Load(int3(b,0)),tex.Load(int3(e.x,b.y,0)),f.x),
                lerp(tex.Load(int3(b.x,e.y,0)),tex.Load(int3(e,0)),f.x),f.y);
}
float3 fusion(int2 p) {
    if(!fusionOn || !historyCount)return 0;
    float2 base=clamp((float2(p)+.5)*float2(iw,ih)/float2(w,h)-.5,0,float2(iw-1,ih-1));
    float2 q=base;float3 center=bilinearColor(Source,base),sum=0;float total=1;
    const float weights[4]={.34,.17,.10,.06};
    [unroll] for(uint step=0;step<4;++step) {
        if(step>=historyCount)break;
        q+=bilinearFlow(Flow[step],q);
        if(any(q<0)||q.x>iw-1||q.y>ih-1||any(!isfinite(q)))break;
        float3 prior=bilinearColor(History[step],q);
        float error=abs(dot(prior-center,float3(.299,.587,.114)));
        float weight=.35*weights[step]*exp(-error/11)*saturate((26-error)/18)*saturate((96-length(q-base))/64);
        sum+=clamp(prior-center,-24,24)*weight; total+=weight;
    }
    return sum/total;
}
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID) {
    if(id.x>=w||id.y>=h)return;int2 p=id.xy;
    if(stage==0) {ColorOut[p]=float4(cubicGuide(p),1);return;}
    if(stage==1) {
        float sx=gray(p+int2(1,-1))+2*gray(p+int2(1,0))+gray(p+int2(1,1))
                -gray(p+int2(-1,-1))-2*gray(p+int2(-1,0))-gray(p+int2(-1,1));
        MaskOut[p]=saturate((abs(sx)/8-.5)/4);return;
    }
    if(stage==2) {
        float sum=0,den=0;for(int i=-10;i<=10;++i){float k=gaussian(i);sum+=k*Mask[reflectp(p+int2(i,0))];den+=k;}
        MaskOut[p]=sum/den;return;
    }
    float3 result=Sr[p].rgb*255+fusion(p);
    if(guardOn) {
        float sum=0,den=0;for(int i=-10;i<=10;++i){float k=gaussian(i);sum+=k*Horizontal[reflectp(p+int2(0,i))];den+=k;}
        result=lerp(result,Guide[p].rgb*255,min(sum/den*.90,.90));
    }
    // Explicit byte quantization: CPU guard casts clipped float32 to uint8.
    ColorOut[p]=float4(floor(clamp(result,0,255)+.00001)/255,1);
}
