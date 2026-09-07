// GPU Block Lite: resize the luma pair for the shared fixed_block_v1 search.
// The search kernel itself remains surface_gpu_motion_tile; this pass only
// changes its sampling grid and never touches the color/XeSS input resolution.
Texture2D<float> PreviousLuma : register(t0);
Texture2D<float> CurrentLuma : register(t1);
RWTexture2D<float> PreviousLite : register(u0);
RWTexture2D<float> CurrentLite : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
};

int2 cp(int2 p) {
    return clamp(p, int2(0, 0), int2(int(inputWidth) - 1, int(inputHeight) - 1));
}

float sample_bilinear(Texture2D<float> image, float2 source) {
    const int2 base = int2(floor(source));
    const float2 f = frac(source);
    const float a = image.Load(int3(cp(base + int2(0, 0)), 0));
    const float b = image.Load(int3(cp(base + int2(1, 0)), 0));
    const float c = image.Load(int3(cp(base + int2(0, 1)), 0));
    const float d = image.Load(int3(cp(base + int2(1, 1)), 0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    const float2 scale = float2(inputWidth / float(width),
                                inputHeight / float(height));
    const float2 source = (float2(tid.xy) + 0.5f) * scale - 0.5f;
    PreviousLite[tid.xy] = sample_bilinear(PreviousLuma, source);
    CurrentLite[tid.xy] = sample_bilinear(CurrentLuma, source);
}
