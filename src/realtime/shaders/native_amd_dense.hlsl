// Official signed SOURCE-pixel vectors at ceil(source/8) block centers.
// Bilinear dense expansion only: never negate a forward vector as a reverse.
Texture2D<int2> Grid:register(t0);
RWTexture2D<float2> Dense:register(u0);
cbuffer Params:register(b0){uint width,height,gridWidth,gridHeight;};
float2 sampleGrid(int2 p){return float2(Grid.Load(int3(clamp(p,int2(0,0),int2(gridWidth-1,gridHeight-1)),0)));}
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID){
    if(id.x>=width||id.y>=height)return;
    float2 p=(float2(id.xy)+.5)/8-.5;int2 q=int2(floor(p));float2 a=frac(p);
    Dense[id.xy]=lerp(lerp(sampleGrid(q),sampleGrid(q+int2(1,0)),a.x),
                     lerp(sampleGrid(q+int2(0,1)),sampleGrid(q+1),a.x),a.y);
}
