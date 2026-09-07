Texture2D<float> FrameA : register(t0);
Texture2D<float> FrameB : register(t1);
RWTexture2D<float2> ForwardOut : register(u0);
RWTexture2D<float2> BackwardOut : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint scenario; int shiftX;
    int shiftY; float angle; float centerX; float centerY;
};
int2 cp(int2 p) { return clamp(p, int2(0, 0), int2(width - 1, height - 1)); }
float patch_cost(Texture2D<float> a, Texture2D<float> b, int2 p, int2 q) {
    float total = 0.0f;
    [unroll] for (int py = -1; py <= 1; ++py) [unroll] for (int px = -1; px <= 1; ++px) {
        int2 ap = cp(p + int2(px, py)); int2 bp = cp(q + int2(px, py));
        total += abs(a.Load(int3(ap, 0)) - b.Load(int3(bp, 0)));
    }
    return total;
}
float2 search_f(Texture2D<float> a, Texture2D<float> b, int2 p) {
    float best = 1e20f; int2 v = 0;
    // The coarsest level covers large motion; with two 2x reductions this
    // radius covers more than +/-32 pixels at the input resolution.
    [loop] for (int y = -8; y <= 8; ++y) [loop] for (int x = -8; x <= 8; ++x) {
        int2 q = cp(p + int2(x, y)); float e = patch_cost(a, b, p, q);
        if (e < best) { best = e; v = int2(x, y); }
    }
    return float2(v);
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(tid.xy); ForwardOut[p] = search_f(FrameA, FrameB, p); BackwardOut[p] = search_f(FrameB, FrameA, p);
}
