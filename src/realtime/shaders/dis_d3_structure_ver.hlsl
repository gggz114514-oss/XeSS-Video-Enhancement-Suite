// Z2 exact OpenCV precomputeStructureTensor, vertical pass.
// One thread per sparse column: running sums over aux rows with float
// deltas, stored at every patch_stride in cv2 order.
Texture2D<float> XX : register(t0);
#ifndef DIS_TENSOR_STRIDE
#define DIS_TENSOR_STRIDE 4
#endif
Texture2D<float> YY : register(t1);
Texture2D<float> XY : register(t2);
Texture2D<float> XS : register(t3);
Texture2D<float> YS : register(t4);
RWTexture2D<float> OutXX : register(u0);
RWTexture2D<float> OutYY : register(u1);
RWTexture2D<float> OutXY : register(u2);
RWTexture2D<float> OutXS : register(u3);
RWTexture2D<float> OutYS : register(u4);
cbuffer Params : register(b0) { uint width; uint height; uint ws; uint hs; };

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= ws) return;
    float sxx = 0.0, syy = 0.0, sxy = 0.0, sx = 0.0, sy = 0.0;
    for (uint i = 0; i < 8; ++i)
    {
        sxx += XX.Load(int3(j, i, 0));
        syy += YY.Load(int3(j, i, 0));
        sxy += XY.Load(int3(j, i, 0));
        sx += XS.Load(int3(j, i, 0));
        sy += YS.Load(int3(j, i, 0));
    }
    OutXX[uint2(j, 0)] = sxx;
    OutYY[uint2(j, 0)] = syy;
    OutXY[uint2(j, 0)] = sxy;
    OutXS[uint2(j, 0)] = sx;
    OutYS[uint2(j, 0)] = sy;
    uint is = 1;
    for (uint i = 8; i < height; ++i)
    {
        sxx += (XX.Load(int3(j, i, 0)) - XX.Load(int3(j, i - 8, 0)));
        syy += (YY.Load(int3(j, i, 0)) - YY.Load(int3(j, i - 8, 0)));
        sxy += (XY.Load(int3(j, i, 0)) - XY.Load(int3(j, i - 8, 0)));
        sx += (XS.Load(int3(j, i, 0)) - XS.Load(int3(j, i - 8, 0)));
        sy += (YS.Load(int3(j, i, 0)) - YS.Load(int3(j, i - 8, 0)));
        if ((i - 7) % DIS_TENSOR_STRIDE == 0)
        {
            OutXX[uint2(j, is)] = sxx;
            OutYY[uint2(j, is)] = syy;
            OutXY[uint2(j, is)] = sxy;
            OutXS[uint2(j, is)] = sx;
            OutYS[uint2(j, is)] = sy;
            is++;
        }
    }
}
