// Gate D3 correctness kernel. It follows OpenCV 5.0.0 DIS's two directional
// propagation sweeps and inverse-search gradient descent. A single invocation
// deliberately serializes the sparse grid so spatial candidates are exactly
// deterministic; D4 may replace this with ordered row/column passes after the
// values are closed against the CPU reference.
Texture2D<float> I0 : register(t0);
Texture2D<float> I1 : register(t1);
Texture2D<int2> Gradient : register(t2);
Texture2D<float> XX : register(t3);
Texture2D<float> YY : register(t4);
Texture2D<float> XY : register(t5);
Texture2D<float> XSum : register(t6);
Texture2D<float> YSum : register(t7);
RWTexture2D<float2> Sparse : register(u0);
cbuffer Params : register(b0) { uint width; uint height; uint ws; uint hs; uint patch; uint stride; uint gdIterations; uint spatial; };

float sample_i1(float x, float y)
{
    x = clamp(x, 0.0, (float)width - 1.0); y = clamp(y, 0.0, (float)height - 1.0);
    int x0 = (int)floor(x), y0 = (int)floor(y);
    int x1 = min(x0 + 1, (int)width - 1), y1 = min(y0 + 1, (int)height - 1);
    float fx = x - (float)x0, fy = y - (float)y0;
    float a = I1.Load(int3(x0, y0, 0)), b = I1.Load(int3(x1, y0, 0));
    float c = I1.Load(int3(x0, y1, 0)), d = I1.Load(int3(x1, y1, 0));
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
}

float patch_ssd(uint px, uint py, float2 u)
{
    float sum = 0.0, sq = 0.0;
    for (uint y = 0; y < 8; ++y) for (uint x = 0; x < 8; ++x) {
        float d = sample_i1((float)(px + x) + u.x, (float)(py + y) + u.y)
                - I0.Load(int3(px + x, py + y, 0));
        sum += d; sq += d * d;
    }
    return sq - sum * sum / 64.0;
}

float4 patch_residual(uint px, uint py, float2 u)
{
    float sum = 0.0, sq = 0.0, dxsum = 0.0, dysum = 0.0;
    for (uint y = 0; y < 8; ++y) for (uint x = 0; x < 8; ++x) {
        float d = sample_i1((float)(px + x) + u.x, (float)(py + y) + u.y)
                - I0.Load(int3(px + x, py + y, 0));
        int2 g = Gradient.Load(int3(px + x, py + y, 0));
        sum += d; sq += d * d; dxsum += d * (float)g.x; dysum += d * (float)g.y;
    }
    return float4(sq - sum * sum / 64.0, dxsum - sum * XSum.Load(int3(px / 4, py / 4, 0)) / 64.0,
                  dysum - sum * YSum.Load(int3(px / 4, py / 4, 0)) / 64.0, 0.0);
}

float2 refine(uint px, uint py, float2 seed)
{
    uint si = py / stride, sj = px / stride;
    float xx = XX.Load(int3(sj, si, 0)), yy = YY.Load(int3(sj, si, 0));
    float xy = XY.Load(int3(sj, si, 0));
    float det = xx * yy - xy * xy;
    float inv = 1.0 / ((abs(det) < 0.001) ? 0.001 : det);
    float inv11 = yy * inv, inv12 = -xy * inv, inv22 = xx * inv;
    float2 u = seed; float previous = 1e10;
    for (uint t = 0; t < gdIterations; ++t) {
        float4 r = patch_residual(px, py, u);
        float2 delta = float2(inv11 * r.y + inv12 * r.z, inv12 * r.y + inv22 * r.z);
        u -= delta;
        if (r.x >= previous) break;
        previous = r.x;
    }
    float2 delta = u - seed;
    return (dot(delta, delta) <= 64.0) ? u : seed;
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x != 0 || id.y != 0) return;
    // OpenCV's Fast DIS uses two propagation iterations when spatial
    // propagation is enabled, with eight gradient steps in each direction.
    for (uint iter = 0; iter < 2; ++iter) {
        if (iter == 0) {
            for (uint si = 0; si < hs; ++si) for (uint sj = 0; sj < ws; ++sj) {
                uint px = sj * stride, py = si * stride; float2 u = float2(0, 0);
                float best = patch_ssd(px, py, u);
                if (sj > 0) { float2 c = Sparse.Load(int2(sj - 1, si)); float s = patch_ssd(px, py, c); if (s < best) { best = s; u = c; } }
                if (si > 0) { float2 c = Sparse.Load(int2(sj, si - 1)); float s = patch_ssd(px, py, c); if (s < best) { best = s; u = c; } }
                Sparse[uint2(sj, si)] = refine(px, py, u);
            }
        } else {
            for (int si = (int)hs - 1; si >= 0; --si) for (int sj = (int)ws - 1; sj >= 0; --sj) {
                uint px = (uint)sj * stride, py = (uint)si * stride; float2 u = Sparse.Load(int2(sj, si));
                float best = patch_ssd(px, py, u);
                if (sj + 1 < (int)ws) { float2 c = Sparse.Load(int2(sj + 1, si)); float s = patch_ssd(px, py, c); if (s < best) { best = s; u = c; } }
                if (si + 1 < (int)hs) { float2 c = Sparse.Load(int2(sj, si + 1)); float s = patch_ssd(px, py, c); if (s < best) { best = s; u = c; } }
                Sparse[uint2(sj, si)] = refine(px, py, u);
            }
        }
    }
}
