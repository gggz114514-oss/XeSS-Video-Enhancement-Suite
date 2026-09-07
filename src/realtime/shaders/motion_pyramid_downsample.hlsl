Texture2D<float> FrameA : register(t0);
Texture2D<float> FrameB : register(t1);
RWTexture2D<float> NextA : register(u0);
RWTexture2D<float> NextB : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint sourceWidth; uint sourceHeight;
    uint unused0; uint unused1; uint unused2; uint unused3;
};

int2 cp(int2 p) { return clamp(p, int2(0, 0), int2(sourceWidth - 1, sourceHeight - 1)); }
float avg(Texture2D<float> f, int2 p) {
    return 0.25f * (f.Load(int3(cp(p * 2 + int2(0, 0)), 0)) +
                    f.Load(int3(cp(p * 2 + int2(1, 0)), 0)) +
                    f.Load(int3(cp(p * 2 + int2(0, 1)), 0)) +
                    f.Load(int3(cp(p * 2 + int2(1, 1)), 0)));
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(tid.xy); NextA[p] = avg(FrameA, p); NextB[p] = avg(FrameB, p);
}
