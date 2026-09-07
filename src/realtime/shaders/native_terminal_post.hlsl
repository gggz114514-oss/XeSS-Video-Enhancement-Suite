// Terminal-only fixed sharpening: one dispatch per emitted frame, after FG.
// OpenCV BORDER_REFLECT_101 boundaries. No temporal filter or second motion pass.
Texture2D<float4> Input : register(t0);
RWTexture2D<float4> Output : register(u0);
cbuffer Params:register(b0) {uint w;uint h;uint strength100;uint reserved;uint4 unused;};
int reflect101(int p,int n) {if(n<=1)return 0;return p<0 ? -p : p>=n ? 2*n-2-p : p;}
float3 sample(int2 p) {p=int2(reflect101(p.x,w),reflect101(p.y,h));return Input.Load(int3(p,0)).rgb*255;}
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID) {
    if(id.x>=w||id.y>=h)return;
    int2 p=id.xy;float3 center=sample(p),blur=0;float lo=255,hi=0;
    const float3 luma=float3(.299,.587,.114);
    const float weights[3][3]={{.077847,.123317,.077847},{.123317,.195346,.123317},{.077847,.123317,.077847}};
    for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
        float3 c=sample(p+int2(x,y));float v=dot(c,luma);lo=min(lo,v);hi=max(hi,v);blur+=weights[y+1][x+1]*c;
    }
    float contrast=min(saturate((hi-lo-72)/80),.65);
    float strength=min((1-contrast)*strength100/100.0*1.65,1.65);
    Output[p]=float4(saturate((center+clamp(center-blur,-24,24)*strength)/255),1);
}
