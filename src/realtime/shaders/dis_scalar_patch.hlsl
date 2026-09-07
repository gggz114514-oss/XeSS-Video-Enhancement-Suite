// Z2c isolation level 1: single-thread SCALAR patch cost with a full
// per-sample trace.  One thread replicates the OpenCV sequential
// computeSSDMeanNorm path for a single patch and a single candidate and
// records every intermediate to a 64x16 R32 trace texture:
//   row = sample (r*8+c); cols: r, c, a, b, c_tap, d_tap, i0, diff,
//   w00, w01, w10, w11, sum_diff_before, sum_sq_before, sum_diff_after,
//   sum_sq_after.  Row 64+0: (i_I1, j_I1, di, dj, sum_diff_r, sum_sq_r,
//   ssd, 0).
Texture2D<float> I0 : register(t0);
Texture2D<float> I1 : register(t1);
RWTexture2D<float2> Trace : register(u0);

cbuffer Params : register(b0)
{
    uint width;
    uint height;
    uint px;
    uint py;
    uint uxBits;
    uint uyBits;
};

[numthreads(1, 1, 1)]
void main()
{
    float ux = 0.0;
    float uy = 0.0;
    precise float i_I1 = min(max((float)py + uy + 16.0, 9.0), (float)height + 15.0);
    precise float j_I1 = min(max((float)px + ux + 16.0, 9.0), (float)width + 15.0);
    precise float di = i_I1 - floor(i_I1);
    precise float dj = j_I1 - floor(j_I1);
    precise float w11 = di * dj;
    precise float w10 = di * (1.0 - dj);
    precise float w01 = (1.0 - di) * dj;
    precise float w00 = (1.0 - di) * (1.0 - dj);
    int base_row = (int)floor(i_I1);
    int base_col = (int)floor(j_I1);
    float sum_diff = 0.0;
    float sum_sq = 0.0;
    for (uint r = 0; r < 8; ++r)
    {
        for (uint k = 0; k < 8; ++k)
        {
            int er = base_row + (int)r;
            int ec = base_col + (int)k;
            int r0 = min(max(er - 16, 0), (int)height - 1);
            int r1 = min(max(er + 1 - 16, 0), (int)height - 1);
            int c0 = min(max(ec - 16, 0), (int)width - 1);
            int c1 = min(max(ec + 1 - 16, 0), (int)width - 1);
            float a = I1.Load(int3(c0, r0, 0));
            float b = I1.Load(int3(c1, r0, 0));
            float cc = I1.Load(int3(c0, r1, 0));
            float d = I1.Load(int3(c1, r1, 0));
            float i0 = I0.Load(int3(px + k, py + (int)r, 0));
            precise float t = w00 * a;
            t = t + w01 * b;
            t = t + w10 * cc;
            t = t + w11 * d;
            precise float diff = t - i0;
            float sd_before = sum_diff;
            float sq_before = sum_sq;
            sum_diff = sum_diff + diff;
            sum_sq = sum_sq + diff * diff;
            uint row = r * 8 + k;
            // 16 record values as 8 float2 cells at (x=0..7, y=row)
            Trace[uint2(0, row)] = float2((float)r, (float)k);
            Trace[uint2(1, row)] = float2(a, b);
            Trace[uint2(2, row)] = float2(cc, d);
            Trace[uint2(3, row)] = float2(i0, diff);
            Trace[uint2(4, row)] = float2(w00, w01);
            Trace[uint2(5, row)] = float2(w10, w11);
            Trace[uint2(6, row)] = float2(sd_before, sq_before);
            Trace[uint2(7, row)] = float2(sum_diff, sum_sq);
        }
    }
    // final row at y=46 (spare), x=0..7
    Trace[uint2(0, 46)] = float2(i_I1, j_I1);
    Trace[uint2(1, 46)] = float2(di, dj);
    Trace[uint2(2, 46)] = float2(sum_diff, sum_sq);
    Trace[uint2(3, 46)] = float2(sum_sq - sum_diff * sum_diff / 64.0, (float)px);
    Trace[uint2(4, 46)] = float2((float)py, 0.0);
}
