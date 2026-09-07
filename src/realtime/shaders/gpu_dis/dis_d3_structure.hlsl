// Gate D3 diagnostic: exact OpenCV DIS patch-grid structure tensor.
Texture2D<int2> Gradient : register(t0);
RWTexture2D<float> XX : register(u0);
RWTexture2D<float> YY : register(u1);
RWTexture2D<float> XY : register(u2);
RWTexture2D<float> XSum : register(u3);
RWTexture2D<float> YSum : register(u4);
cbuffer Params : register(b0) { uint width; uint height; uint ws; uint hs; };

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= ws || id.y >= hs) return;
    uint x0 = id.x * 4, y0 = id.y * 4;
    float xx = 0.0, yy = 0.0, xy = 0.0, xs = 0.0, ys = 0.0;
    for (uint y = 0; y < 8; ++y) for (uint x = 0; x < 8; ++x) {
        int2 g = Gradient.Load(int3(min(x0 + x, width - 1), min(y0 + y, height - 1), 0));
        float fx = (float)g.x, fy = (float)g.y;
        xx += fx * fx; yy += fy * fy; xy += fx * fy; xs += fx; ys += fy;
    }
    XX[id.xy] = xx; YY[id.xy] = yy; XY[id.xy] = xy; XSum[id.xy] = xs; YSum[id.xy] = ys;
}
