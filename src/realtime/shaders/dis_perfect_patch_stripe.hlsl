// GPU DIS exact patch inverse search (Z2c).
//
// Exact OpenCV 5.0.0 semantics per candidate and per gradient-descent step:
//   - bilinear patch diff ((w00*a + w01*b) + w10*c) + w11*d - i0, strictly
//     left-associative float32 (precise guards reassociation/FMA);
//   - mean-normalization sums accumulated in 4 lanes (lane k over samples
//     (r, k) and (r, k+4) for r = 0..7, sequential float32 adds), reduced as
//     (v0+v2) + (v1+v3)  (OpenCV shift-tree v_reduce_sum);
//   - candidate evaluation order: seed, side (scan-order neighbor), perp;
//     strict '<' replaces the best;
//   - gradient descent: invH from the separable structure tensor, 8 inner
//     iterations, break when SSD >= previous SSD (after the update), final
//     acceptance |delta|^2 <= 64;
//   - warp coordinate clamp on the patch top-left, replicate border for the
//     patch samples (equivalent to OpenCV's I1_ext access).
//
// Scheduling: one workgroup processes 16 patches (64 threads, 4 lanes each);
// the host dispatches one anti-diagonal per dispatch as before (wavefront
// same-pass propagation semantics).
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
    uint direction;  // 0 = forward (left/up), 1 = backward (right/down)
    uint rowSel;     // stripe row handled by this dispatch (cv 8-stripe)
    uint stripeBits; // lo (bits 0-7) | hi (bits 8-15) | jsSel (bits 16-23)
};

uint stripe_lo() { return stripeBits & 0xFFu; }
uint stripe_hi() { return (stripeBits >> 8) & 0xFFu; }
uint js_sel() { return (stripeBits >> 16) & 0xFFu; }

groupshared float gShare[16][4];  // per-patch reduced lane scalars

// Exact mean-normalized patch residual for one candidate position.
// Returns reduced scalars (sd, sq, gx, gy) for the four lanes.
void patch_residual(uint px, uint py, float ux, float uy, uint lane,
                    out float out_sd, out float out_sq,
                    out float out_gx, out float out_gy)
{
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

    precise float sd = 0.0;
    precise float sq = 0.0;
    precise float gx = 0.0;
    precise float gy = 0.0;
    uint k = lane;
    for (uint r = 0; r < 8; ++r)
    {
        precise float d_l = 0.0;
        precise float d_r = 0.0;
        // two samples per row per lane: k and k+4
        for (uint q = 0; q < 2; ++q)
        {
            uint c = k + q * 4;
            // replicate-border taps of I1 (I1_ext equivalent)
            int er = base_row + (int)r;
            int ec = base_col + (int)c;
            int r0 = min(max(er - 16, 0), (int)height - 1);
            int r1 = min(max(er + 1 - 16, 0), (int)height - 1);
            int c0 = min(max(ec - 16, 0), (int)width - 1);
            int c1 = min(max(ec + 1 - 16, 0), (int)width - 1);
            float a = I1.Load(int3(c0, r0, 0));
            float b = I1.Load(int3(c1, r0, 0));
            float cc = I1.Load(int3(c0, r1, 0));
            float d = I1.Load(int3(c1, r1, 0));
            precise float t = w00 * a;
            t = t + w01 * b;
            t = t + w10 * cc;
            t = t + w11 * d;
            precise float diff = t - I0.Load(int3(px + c, py + (int)r, 0));
            if (q == 0)
                d_l = diff;
            else
                d_r = diff;
        }
        precise float t_sd = d_l + d_r;
        sd = sd + t_sd;
        precise float t_sq = d_l * d_l + d_r * d_r;
        sq = sq + t_sq;
        int2 g0 = Gradient.Load(int3(px + k, py + (int)r, 0));
        int2 g1 = Gradient.Load(int3(px + k + 4, py + (int)r, 0));
        precise float t_gx = d_l * (float)g0.x + d_r * (float)g1.x;
        gx = gx + t_gx;
        precise float t_gy = d_l * (float)g0.y + d_r * (float)g1.y;
        gy = gy + t_gy;
    }
    out_sd = sd;
    out_sq = sq;
    out_gx = gx;
    out_gy = gy;
}

// (v0+v2) + (v1+v3) across the four lanes of one patch; result on lane 0.
// gs_index must be float: a uint signature silently truncated negative lane
// sums (hardware fptoui clamps them to 0), corrupting every sd reduction.
// Callers insert a GroupMemoryBarrier after reading gShare so the phase-1
// writes of the next reduce_lanes call cannot race the previous read.
void reduce_lanes(uint patch, uint lane, float gs_index)
{
    gShare[patch][lane] = gs_index;
    GroupMemoryBarrierWithGroupSync();
    if (lane < 2)
    {
        gShare[patch][lane] = gShare[patch][lane] + gShare[patch][lane + 2];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0)
    {
        gShare[patch][2] = gShare[patch][0] + gShare[patch][1];
    }
    GroupMemoryBarrierWithGroupSync();
}

float candidate_ssd(uint px, uint py, float2 u, uint patch, uint lane)
{
    float sd, sq, gx, gy;
    patch_residual(px, py, u.x, u.y, lane, sd, sq, gx, gy);
    reduce_lanes(patch, lane, sd);
    float s_diff = gShare[patch][2];
    GroupMemoryBarrierWithGroupSync();
    reduce_lanes(patch, lane, sq);
    float s_sq = gShare[patch][2];
    GroupMemoryBarrierWithGroupSync();
    float n = 64.0;
    precise float ssd = s_sq - s_diff * s_diff / n;
    return ssd;
}

void candidate_gd(uint px, uint py, float2 seed, uint si, uint sj, uint patch, uint lane,
                  out float2 out_flow, out bool out_ok)
{
    float xx = XX.Load(int3(sj, si, 0));
    float yy = YY.Load(int3(sj, si, 0));
    float xy = XY.Load(int3(sj, si, 0));
    precise float detH = xx * yy - xy * xy;
    if (abs(detH) < 0.001) detH = 0.001;
    precise float inv11 = yy / detH;
    precise float inv12 = -xy / detH;
    precise float inv22 = xx / detH;
    float x_grad = XSum.Load(int3(sj, si, 0));
    float y_grad = YSum.Load(int3(sj, si, 0));
    float2 cur = seed;
    float prev_ssd = 1e10;
    [loop]
    for (uint t = 0; t < gdIterations; ++t)
    {
        float sd, sq, gx, gy;
        patch_residual(px, py, cur.x, cur.y, lane, sd, sq, gx, gy);
        reduce_lanes(patch, lane, sd);
        float s_diff = gShare[patch][2];
        GroupMemoryBarrierWithGroupSync();
        reduce_lanes(patch, lane, sq);
        float s_sq = gShare[patch][2];
        GroupMemoryBarrierWithGroupSync();
        reduce_lanes(patch, lane, gx);
        float s_gx = gShare[patch][2];
        GroupMemoryBarrierWithGroupSync();
        reduce_lanes(patch, lane, gy);
        float s_gy = gShare[patch][2];
        GroupMemoryBarrierWithGroupSync();
        float n = 64.0;
        precise float ssd = s_sq - s_diff * s_diff / n;
        precise float dUx = s_gx - s_diff * x_grad / n;
        precise float dUy = s_gy - s_diff * y_grad / n;
        precise float dx = inv11 * dUx + inv12 * dUy;
        precise float dy = inv12 * dUx + inv22 * dUy;
        cur.x = cur.x - dx;
        cur.y = cur.y - dy;
        if (ssd >= prev_ssd) break;
        prev_ssd = ssd;
    }
    float2 delta = cur - seed;
    out_ok = (delta.x * delta.x + delta.y * delta.y) <= (float)(patchSize * patchSize);
    out_flow = cur;
}

// cv 8-stripe coordinate: one dispatch covers stripe row rowSel; ordinal is the
// in-row column (scan order per direction).
uint2 patch_coordinate(uint ordinal)
{
    uint is = rowSel;
    uint js = ordinal;
    if (direction != 0)
        js = sparseWidth - 1 - js;
    return uint2(js, is);
}

[numthreads(64, 1, 1)]
void main(uint3 groupThread : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    uint lane = groupThread.x & 3;
    if (groupThread.x >> 2 != js_sel()) return;
    uint count = sparseWidth;
    uint ordinal = js_sel();
    if (ordinal >= count) return;
    uint2 coord = patch_coordinate(ordinal);
    uint sj = coord.x, si = coord.y;
    if (sj >= sparseWidth || si >= sparseHeight) return;
    uint px = sj * patchStride, py = si * patchStride;

    float2 seed = float2(0.0, 0.0);
    if (direction == 0)
    {
        if (useInitialFlow != 0)
            seed = InitialFlow.Load(int3(px + patchSize / 2, py + patchSize / 2, 0));
        else
            seed = float2(0.0, 0.0);
    }
    else
    {
        seed = Sparse.Load(int3(sj, si, 0));
    }

    float best = candidate_ssd(px, py, seed, patch, lane);
    float2 best_flow = seed;
    GroupMemoryBarrierWithGroupSync();
    if (direction == 0)
    {
        if (sj > 0)
        {
            float2 cand = Sparse.Load(int3(sj - 1, si, 0));
            float score = candidate_ssd(px, py, cand, patch, lane);
            if (score < best) { best = score; best_flow = cand; seed = cand; }
            GroupMemoryBarrierWithGroupSync();
        }
        if (si > stripe_lo())
        {
            float2 cand = Sparse.Load(int3(sj, si - 1, 0));
            float score = candidate_ssd(px, py, cand, patch, lane);
            if (score < best) { best = score; best_flow = cand; seed = cand; }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    else
    {
        if (sj + 1 < sparseWidth)
        {
            float2 cand = Sparse.Load(int3(sj + 1, si, 0));
            float score = candidate_ssd(px, py, cand, patch, lane);
            if (score < best) { best = score; best_flow = cand; seed = cand; }
            GroupMemoryBarrierWithGroupSync();
        }
        if (si + 1 < stripe_hi())
        {
            float2 cand = Sparse.Load(int3(sj, si + 1, 0));
            float score = candidate_ssd(px, py, cand, patch, lane);
            if (score < best) { best = score; best_flow = cand; seed = cand; }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    float2 flow;
    bool ok;
    GroupMemoryBarrierWithGroupSync();
    candidate_gd(px, py, seed, si, sj, patch, lane, flow, ok);
    GroupMemoryBarrierWithGroupSync();
    float2 result = ok ? flow : seed;
    if (lane == 0)
        Sparse[uint2(sj, si)] = result;
}
