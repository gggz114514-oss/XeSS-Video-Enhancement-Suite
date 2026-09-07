// Z2 exact OpenCV precomputeStructureTensor, horizontal pass.
// One thread per image row: running sums over [j-7..j] windows with exact
// integer products/deltas (short*short in int32) added to float accumulators
// sequentially, stored at sparse columns every patch_stride in cv2 order.
Texture2D<int2> Gradient : register(t0);
#ifndef DIS_TENSOR_STRIDE
#define DIS_TENSOR_STRIDE 4
#endif
RWTexture2D<float> XX : register(u0);
RWTexture2D<float> YY : register(u1);
RWTexture2D<float> XY : register(u2);
RWTexture2D<float> XS : register(u3);
RWTexture2D<float> YS : register(u4);
cbuffer Params : register(b0) { uint width; uint height; uint ws; uint hs; };

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= height) return;
    int outputOffset = 0;
    float sxx = 0.0, syy = 0.0, sxy = 0.0, sx = 0.0, sy = 0.0;
    uint js = 1;
    // initial window [0, 7]: exact int products converted to float, sequential
    for (uint j = 0; j < 8; ++j)
    {
        int2 g = Gradient.Load(int3(j, i, 0));
        int x = g.x, y = g.y;
        sxx += (float)(x * x);
        syy += (float)(y * y);
        sxy += (float)(x * y);
        sx += (float)x;
        sy += (float)y;
    }
    XX[uint2(0, i)] = sxx;
    YY[uint2(0, i)] = syy;
    XY[uint2(0, i)] = sxy;
    XS[uint2(0, i)] = sx;
    YS[uint2(0, i)] = sy;
    // sliding windows with int-exact deltas, cv2 store condition (j-7)%4==0
    for (uint j = 8; j < width; ++j)
    {
        int2 g = Gradient.Load(int3(j, i, 0));
        int2 gp = Gradient.Load(int3(j - 8, i, 0));
        int x = g.x, y = g.y, xp = gp.x, yp = gp.y;
        sxx += (float)(x * x - xp * xp);
        syy += (float)(y * y - yp * yp);
        sxy += (float)(x * y - xp * yp);
        sx += (float)(x - xp);
        sy += (float)(y - yp);
        if ((j - 7) % DIS_TENSOR_STRIDE == 0)
        {
            XX[uint2(js, i)] = sxx;
            YY[uint2(js, i)] = syy;
            XY[uint2(js, i)] = sxy;
            XS[uint2(js, i)] = sx;
            YS[uint2(js, i)] = sy;
            js++;
        }
    }
}
