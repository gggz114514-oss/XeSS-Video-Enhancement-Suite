// OpenCV generic CV_32FC2 INTER_LINEAR: float coordinate table, separable
// unfused weighted sums. Different from IPP C1 inter-level interpolation.
Texture2D<float2> Source : register(t0);
RWTexture2D<float2> Destination : register(u0);
cbuffer Params : register(b0) { uint srcWidth; uint srcHeight; uint dstWidth; uint dstHeight; uint scale; };
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dstWidth || id.y >= dstHeight) return;
    precise double dx = ((double)id.x + 0.5) * (double)srcWidth / (double)dstWidth - 0.5;
    precise double dy = ((double)id.y + 0.5) * (double)srcHeight / (double)dstHeight - 0.5;
    precise float px = (float)dx, py = (float)dy;
    int ix = (int)floor(px), iy = (int)floor(py);
    precise float fx = px - (float)ix, fy = py - (float)iy;
    if (ix < 0 || ix >= (int)srcWidth - 1) fx = 0;
    int x0 = clamp(ix,0,(int)srcWidth-1), x1 = clamp(ix+1,0,(int)srcWidth-1);
    int y0 = clamp(iy,0,(int)srcHeight-1), y1 = clamp(iy+1,0,(int)srcHeight-1);
    float2 a=Source.Load(int3(x0,y0,0)), b=Source.Load(int3(x1,y0,0));
    float2 c=Source.Load(int3(x0,y1,0)), d=Source.Load(int3(x1,y1,0));
    precise float2 top=a*(1-fx)+b*fx, bottom=c*(1-fx)+d*fx;
    precise float2 value=top*(1-fy)+bottom*fy;
    Destination[id.xy]=value*(float)scale;
}
