// GPU DIS perfect-port P1: OpenCV 5.0.0 Fast patch inverse search.
//
// One 8x8 patch is one workgroup.  The 64 lanes evaluate one sample each and
// reduce mean-normalized SSD/Jacobian terms through groupshared trees.  The
// host dispatches one anti-diagonal at a time; GroupID.x is the patch ordinal
// on that diagonal and Params.diagonal selects its coordinates.  This keeps
// OpenCV's left/up and right/down dependency order without a serialized full
// grid scan.

Texture2D<float> I0 : register(t0);
Texture2D<float> I1 : register(t1);
Texture2D<int2> Gradient : register(t2);
Texture2D<float> XX : register(t3);
Texture2D<float> YY : register(t4);
Texture2D<float> XY : register(t5);
Texture2D<float> XSum : register(t6);
Texture2D<float> YSum : register(t7);
Texture2D<float2> InitialFlow : register(t8);
RWTexture2D<float2> Sparse : register(u0);

cbuffer Params : register(b0)
{
    uint width;
    uint height;
    uint sparseWidth;
    uint sparseHeight;
    uint patchSize;
    uint patchStride;
    uint gdIterations;
    uint direction; // 0 = forward left/up, 1 = backward right/down
    uint diagonal;
    uint useInitialFlow;
};

groupshared float gSum[64];
groupshared float gSq[64];
groupshared float gDx[64];
groupshared float gDy[64];
groupshared float gResult[4];

float sample_previous(float2 p)
{
    // OpenCV samples I1_ext with a 16-pixel replicate border and clamps the
    // patch coordinate to [-patch+1,width-1].  Clamping the original image
    // coordinate is equivalent because every outside border texel repeats.
    p.x = clamp(p.x, 0.0, (float)width - 1.0);
    p.y = clamp(p.y, 0.0, (float)height - 1.0);
    int2 lo = int2(floor(p));
    int2 hi = min(lo + 1, int2((int)width - 1, (int)height - 1));
    float2 f = p - (float2)lo;
    float a = I1.Load(int3(lo, 0));
    float b = I1.Load(int3(int2(hi.x, lo.y), 0));
    float c = I1.Load(int3(int2(lo.x, hi.y), 0));
    float d = I1.Load(int3(hi, 0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

void reduce_pair(inout float a, inout float b, uint lane)
{
    gSum[lane] = a;
    gSq[lane] = b;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 32; offset != 0; offset >>= 1)
    {
        if (lane < offset)
        {
            gSum[lane] += gSum[lane + offset];
            gSq[lane] += gSq[lane + offset];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    a = gSum[0];
    b = gSq[0];
}

float candidate_ssd(uint px, uint py, float2 u, uint lane)
{
    uint tx = lane & 7, ty = lane >> 3;
    float2 q = float2((float)px + (float)tx, (float)py + (float)ty) + u;
    float diff = sample_previous(q) - I0.Load(int3(px + tx, py + ty, 0));
    float sum = diff, sq = diff * diff;
    reduce_pair(sum, sq, lane);
    if (lane == 0) gResult[0] = sq - sum * sum / 64.0;
    GroupMemoryBarrierWithGroupSync();
    return gResult[0];
}

float4 patch_residual(uint px, uint py, float2 u, float2 grad_sum, uint lane)
{
    uint tx = lane & 7, ty = lane >> 3;
    float2 q = float2((float)px + (float)tx, (float)py + (float)ty) + u;
    float diff = sample_previous(q) - I0.Load(int3(px + tx, py + ty, 0));
    int2 grad = Gradient.Load(int3(px + tx, py + ty, 0));
    gSum[lane] = diff;
    gSq[lane] = diff * diff;
    gDx[lane] = diff * (float)grad.x;
    gDy[lane] = diff * (float)grad.y;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 32; offset != 0; offset >>= 1)
    {
        if (lane < offset)
        {
            gSum[lane] += gSum[lane + offset];
            gSq[lane] += gSq[lane + offset];
            gDx[lane] += gDx[lane + offset];
            gDy[lane] += gDy[lane + offset];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
    {
        float sum = gSum[0], sq = gSq[0];
        gResult[0] = sq - sum * sum / 64.0;
        gResult[1] = gDx[0] - sum * grad_sum.x / 64.0;
        gResult[2] = gDy[0] - sum * grad_sum.y / 64.0;
    }
    GroupMemoryBarrierWithGroupSync();
    return float4(gResult[0], gResult[1], gResult[2], 0.0);
}

float2 patch_gd(uint px, uint py, float2 seed, uint si, uint sj, uint lane)
{
    float xx = XX.Load(int3(sj, si, 0));
    float yy = YY.Load(int3(sj, si, 0));
    float xy = XY.Load(int3(sj, si, 0));
    float det = xx * yy - xy * xy;
    float inv = 1.0 / ((abs(det) < 0.001) ? 0.001 : det);
    float inv11 = yy * inv, inv12 = -xy * inv, inv22 = xx * inv;
    float2 grad_sum = float2(XSum.Load(int3(sj, si, 0)), YSum.Load(int3(sj, si, 0)));
    float2 u = seed;
    float previous_ssd = 1e10;
    for (uint t = 0; t < 8; ++t)
    {
        float4 r = patch_residual(px, py, u, grad_sum, lane);
        float2 step = float2(inv11 * r.y + inv12 * r.z,
                             inv12 * r.y + inv22 * r.z);
        u -= step;
        if (r.x >= previous_ssd) break;
        previous_ssd = r.x;
    }
    float2 delta = u - seed;
    return (dot(delta, delta) <= (float)(patchSize * patchSize)) ? u : seed;
}

uint2 patch_coordinate(uint ordinal)
{
    uint is = max(0, (int)diagonal - (int)sparseWidth + 1);
    uint js = diagonal - is;
    if (direction != 0)
    {
        uint ris = is;
        uint rjs = js;
        is = sparseHeight - 1 - ris;
        js = sparseWidth - 1 - rjs;
    }
    // For a non-square diagonal the ordinal is the only group in the normal
    // path.  Keep this addition explicit for the indirect/tiled scheduler:
    // the dispatch count is the diagonal length and ordinal walks its rows.
    if (ordinal != 0)
    {
        if (direction == 0)
        {
            is += ordinal;
            js -= ordinal;
        }
        else
        {
            is -= ordinal;
            js += ordinal;
        }
    }
    return uint2(js, is);
}

[numthreads(8, 8, 1)]
void main(uint3 groupThread : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    uint first = max(0, (int)diagonal - (int)sparseWidth + 1);
    uint last = min(diagonal, sparseHeight - 1);
    if (last < first || group.x >= last - first + 1) return;
    uint2 coord = patch_coordinate(group.x);
    uint sj = coord.x, si = coord.y;
    if (sj >= sparseWidth || si >= sparseHeight) return;
    uint px = sj * patchStride, py = si * patchStride;
    uint lane = groupThread.y * 8 + groupThread.x;

    float2 seed = float2(0.0, 0.0);
    if (direction == 0 && useInitialFlow != 0)
        seed = InitialFlow.Load(int3(px + patchSize / 2, py + patchSize / 2, 0));
    if (direction != 0) seed = Sparse.Load(int3(sj, si, 0)).xy;
    float best = candidate_ssd(px, py, seed, lane);

    if (direction == 0)
    {
        if (sj > 0)
        {
            float2 candidate = Sparse.Load(int3(sj - 1, si, 0)).xy;
            float score = candidate_ssd(px, py, candidate, lane);
            if (score < best) { best = score; seed = candidate; }
        }
        if (si > 0)
        {
            float2 candidate = Sparse.Load(int3(sj, si - 1, 0)).xy;
            float score = candidate_ssd(px, py, candidate, lane);
            if (score < best) { best = score; seed = candidate; }
        }
    }
    else
    {
        if (sj + 1 < sparseWidth)
        {
            float2 candidate = Sparse.Load(int3(sj + 1, si, 0)).xy;
            float score = candidate_ssd(px, py, candidate, lane);
            if (score < best) { best = score; seed = candidate; }
        }
        if (si + 1 < sparseHeight)
        {
            float2 candidate = Sparse.Load(int3(sj, si + 1, 0)).xy;
            float score = candidate_ssd(px, py, candidate, lane);
            if (score < best) { best = score; seed = candidate; }
        }
    }

    float2 result = patch_gd(px, py, seed, si, sj, lane);
    if (lane == 0) Sparse[uint2(sj, si)] = result;
}
