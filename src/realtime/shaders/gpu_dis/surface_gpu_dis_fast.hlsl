// Gate D5 resident Fast-DIS motion pass.
// This is the actual DIS patch inverse-search stage (8x8 patch, mean
// normalization, structure tensor and 8 gradient-descent steps), executed
// directly from the oneVPL-converted luma surface. It deliberately has no
// full-frame CPU boundary. The D5 probe uses this per-pixel patch seed while
// D6 owns the multi-level/persistent production scheduler.
Texture2D<float> PreviousLuma : register(t0);
Texture2D<float> CurrentLuma : register(t1);
RWTexture2D<float2> FlowOut : register(u0);
RWTexture2D<float> MaskOut : register(u1);
cbuffer Params : register(b0) { uint width; uint height; uint inputWidth; uint inputHeight; };

int2 cp(int2 p) { return clamp(p, int2(0, 0), int2((int)inputWidth - 1, (int)inputHeight - 1)); }
float cur(int2 p) { return CurrentLuma.Load(int3(cp(p), 0)); }
float prev(int2 p) { return PreviousLuma.Load(int3(cp(p), 0)); }
float sample_prev(float x, float y)
{
    x = clamp(x, 0.0, (float)inputWidth - 1.001); y = clamp(y, 0.0, (float)inputHeight - 1.001);
    int2 p = int2(floor(x), floor(y)); int2 q = cp(p + int2(1, 1)); p = cp(p);
    float2 f = float2(x - floor(x), y - floor(y));
    return lerp(lerp(prev(p), prev(int2(q.x, p.y)), f.x),
                lerp(prev(int2(p.x, q.y)), prev(q), f.x), f.y);
}
float2 sobel(int2 p)
{
    int2 q = cp(p); int x = q.x, y = q.y;
    float gx = -cur(int2(x - 1, y - 1)) - 2.0 * cur(int2(x - 1, y)) - cur(int2(x - 1, y + 1))
             + cur(int2(x + 1, y - 1)) + 2.0 * cur(int2(x + 1, y)) + cur(int2(x + 1, y + 1));
    float gy = -cur(int2(x - 1, y - 1)) - 2.0 * cur(int2(x, y - 1)) - cur(int2(x + 1, y - 1))
             + cur(int2(x - 1, y + 1)) + 2.0 * cur(int2(x, y + 1)) + cur(int2(x + 1, y + 1));
    if (x == 0 || x + 1 >= (int)inputWidth) gx = 0.0;
    if (y == 0 || y + 1 >= (int)inputHeight) gy = 0.0;
    return float2(gx, gy);
}
struct PatchEval { float ssd; float dx; float dy; };
PatchEval eval_patch(int2 top, float2 u, float xsum, float ysum)
{
    float sum = 0.0, sq = 0.0, dxsum = 0.0, dysum = 0.0;
    for (int y = 0; y < 8; ++y) for (int x = 0; x < 8; ++x) {
        int2 p = top + int2(x, y); float d = sample_prev(p.x + u.x, p.y + u.y) - cur(p); float2 g = sobel(p);
        sum += d; sq += d * d; dxsum += d * g.x; dysum += d * g.y;
    }
    PatchEval result; result.ssd = sq - sum * sum / 64.0; result.dx = dxsum - sum * xsum / 64.0; result.dy = dysum - sum * ysum / 64.0; return result;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(min(tid.x * inputWidth / width, inputWidth - 1), min(tid.y * inputHeight / height, inputHeight - 1));
    int2 top = cp(p - int2(3, 3)); top = min(top, int2((int)inputWidth - 8, (int)inputHeight - 8));
    float xx = 0.0, yy = 0.0, xy = 0.0, xsum = 0.0, ysum = 0.0;
    for (int y = 0; y < 8; ++y) for (int x = 0; x < 8; ++x) { float2 g = sobel(top + int2(x, y)); xx += g.x * g.x; yy += g.y * g.y; xy += g.x * g.y; xsum += g.x; ysum += g.y; }
    float det = xx * yy - xy * xy; if (abs(det) < 0.001) det = 0.001;
    float inv11 = yy / det, inv12 = -xy / det, inv22 = xx / det; float2 u = float2(0, 0); float previous_ssd = 1e10;
    for (int t = 0; t < 8; ++t) { PatchEval e = eval_patch(top, u, xsum, ysum); float2 d = float2(inv11 * e.dx + inv12 * e.dy, inv12 * e.dx + inv22 * e.dy); u -= d; if (e.ssd >= previous_ssd) break; previous_ssd = e.ssd; }
    if (dot(u, u) > 64.0) u = float2(0, 0);
    FlowOut[tid.xy] = u * float2(width / (float)inputWidth, height / (float)inputHeight);
    MaskOut[tid.xy] = 0.0;
}
