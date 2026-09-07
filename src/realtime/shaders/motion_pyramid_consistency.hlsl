Texture2D<float2> ForwardFlow : register(t0);
Texture2D<float2> BackwardFlow : register(t1);
Texture2D<float2> ProbeFlow1 : register(t2);
Texture2D<float2> ProbeFlow2 : register(t3);
RWTexture2D<float> ConsistencyMask : register(u0);
RWStructuredBuffer<uint> Summary : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint scenario; int shiftX;
    int shiftY; float angle; float centerX; float centerY;
};
int2 cp(int2 p) { return clamp(p, int2(0, 0), int2(width - 1, height - 1)); }
float2 expected(int2 p) {
    if (scenario == 0) return 0.0f;
    if (scenario == 1 || scenario == 3 || scenario == 4) return float2(-shiftX, -shiftY);
    float c = cos(angle), s = sin(angle); float2 d = float2(p) - float2(centerX, centerY);
    float2 q = float2(centerX, centerY) + float2(c * d.x + s * d.y, -s * d.x + c * d.y);
    return q - float2(p);
}
bool truth_valid(int2 p) {
    if (scenario != 3) return true;
    return !(p.x >= int(width / 3) && p.x < int(width / 2) && p.y >= int(height / 3) && p.y < int(height * 2 / 3));
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(tid.xy); float2 f = ForwardFlow.Load(int3(p, 0)); if (p.x == 8 && p.y == 8) { Summary[10] = asuint(f.x); Summary[11] = asuint(f.y); Summary[12] = asuint(ProbeFlow1.Load(int3(p / 2, 0)).x); Summary[13] = asuint(ProbeFlow1.Load(int3(p / 2, 0)).y); Summary[14] = asuint(ProbeFlow2.Load(int3(p / 4, 0)).x); Summary[15] = asuint(ProbeFlow2.Load(int3(p / 4, 0)).y); } int2 q = cp(int2(round(float2(p) + f)));
    float2 b = BackwardFlow.Load(int3(q, 0)); bool consistent = length(f + b) <= 1.5f; ConsistencyMask[p] = consistent ? 1.0f : 0.0f;
    int border = scenario == 2 ? 8 : 4; if (scenario == 1 || scenario == 3 || scenario == 4) border += max(abs(shiftX), abs(shiftY));
    bool core = p.x >= border && p.y >= border && p.x + border < int(width) && p.y + border < int(height); if (!core) return;
    InterlockedAdd(Summary[0], 1); if (consistent) InterlockedAdd(Summary[2], 1); else InterlockedAdd(Summary[3], 1);
    InterlockedAdd(Summary[8], uint(round(abs(f.x) * 1000.0f))); InterlockedAdd(Summary[9], uint(round(abs(f.y) * 1000.0f)));
    float2 e = abs(f - expected(p)); if (truth_valid(p)) { InterlockedAdd(Summary[1], 1); InterlockedAdd(Summary[4], uint(round(e.x * 1000.0f))); InterlockedAdd(Summary[5], uint(round(e.y * 1000.0f))); InterlockedMax(Summary[6], uint(round((e.x + e.y) * 1000.0f))); if (e.x <= 0.5f && e.y <= 0.5f) InterlockedAdd(Summary[16], 1); } else InterlockedAdd(Summary[7], 1);
}
