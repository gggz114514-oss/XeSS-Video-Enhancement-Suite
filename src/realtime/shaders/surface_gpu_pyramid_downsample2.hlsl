// H3 Fast pyramid half-resolution luma level.
Texture2D<float> PreviousLuma : register(t0);
Texture2D<float> CurrentLuma : register(t1);
RWTexture2D<float> PreviousHalf : register(u0);
RWTexture2D<float> CurrentHalf : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
};

int2 cp(int2 p) {
    return clamp(p, int2(0, 0),
                 int2(int(inputWidth) - 1, int(inputHeight) - 1));
}

float reduce2(Texture2D<float> image, int2 p) {
    return 0.25f * (image.Load(int3(cp(p * 2 + int2(0, 0)), 0)) +
                    image.Load(int3(cp(p * 2 + int2(1, 0)), 0)) +
                    image.Load(int3(cp(p * 2 + int2(0, 1)), 0)) +
                    image.Load(int3(cp(p * 2 + int2(1, 1)), 0)));
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    const int2 p = int2(tid.xy);
    PreviousHalf[p] = reduce2(PreviousLuma, p);
    CurrentHalf[p] = reduce2(CurrentLuma, p);
}
