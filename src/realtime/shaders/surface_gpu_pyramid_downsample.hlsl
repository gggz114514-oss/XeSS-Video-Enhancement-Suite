// H3 Fast pyramid level: exact 4x box reduction from the input luma pair.
Texture2D<float> PreviousLuma : register(t0);
Texture2D<float> CurrentLuma : register(t1);
RWTexture2D<float> PreviousQuarter : register(u0);
RWTexture2D<float> CurrentQuarter : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
};

int2 cp(int2 p) {
    return clamp(p, int2(0, 0),
                 int2(int(inputWidth) - 1, int(inputHeight) - 1));
}

float reduce4(Texture2D<float> image, int2 p) {
    float sum = 0.0f;
    [unroll] for (int y = 0; y < 4; ++y)
        [unroll] for (int x = 0; x < 4; ++x)
            sum += image.Load(int3(cp(p * 4 + int2(x, y)), 0));
    return sum * (1.0f / 16.0f);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    const int2 p = int2(tid.xy);
    PreviousQuarter[p] = reduce4(PreviousLuma, p);
    CurrentQuarter[p] = reduce4(CurrentLuma, p);
}
