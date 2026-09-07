Texture2D<int2> ForwardFlow : register(t0);
Texture2D<int2> BackwardFlow : register(t1);
RWTexture2D<float> ConsistencyMask : register(u0);
RWStructuredBuffer<uint> Summary : register(u1);

cbuffer Params : register(b0) {
    uint width;
    uint height;
    uint scenario;
    int shiftX;
    int shiftY;
    float angle;
    float centerX;
    float centerY;
};

int2 cp(int2 p) { return clamp(p, int2(0, 0), int2(width - 1, height - 1)); }

int2 expected_flow(int2 p) {
    if (scenario == 0) return int2(0, 0);
    if (scenario == 1 || scenario == 3 || scenario == 4) return int2(-shiftX, -shiftY);
    // Scenario 2: current(p) was generated from reference(R^-1(p)); this is
    // the expected current->reference vector for the rotation case.
    float c = cos(angle), s = sin(angle);
    float2 d = float2(p) - float2(centerX, centerY);
    float2 q = float2(centerX, centerY) + float2(c * d.x + s * d.y, -s * d.x + c * d.y);
    return int2(round(q - float2(p)));
}

bool truth_valid(int2 p) {
    // A deliberately synthetic occluder. Pixels in the newly uncovered patch
    // are excluded from EPE but remain in the consistency-mask accounting.
    if (scenario != 3) return true;
    return !(p.x >= int(width / 3) && p.x < int(width / 2) &&
             p.y >= int(height / 3) && p.y < int(height * 2 / 3));
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(tid.xy);
    // The bounded search clamps samples at the image edge. Keep those pixels
    // in the emitted mask, but exclude them from the quality reduction so an
    // artificial border tie cannot masquerade as motion error.
    bool core = p.x >= 4 && p.y >= 4 && p.x + 4 < int(width) && p.y + 4 < int(height);
    int2 f = ForwardFlow.Load(int3(p, 0));
    int2 b = BackwardFlow.Load(int3(cp(p + f), 0));
    int consistency = abs(f.x + b.x) + abs(f.y + b.y);
    bool consistent = consistency <= 1;
    ConsistencyMask[p] = consistent ? 1.0f : 0.0f;
    if (!core) return;
    InterlockedAdd(Summary[0], 1); // pixels
    if (consistent) InterlockedAdd(Summary[2], 1); // consistent pixels
    else InterlockedAdd(Summary[3], 1); // inconsistent pixels
    int2 e = abs(f - expected_flow(p));
    if (truth_valid(p)) {
        InterlockedAdd(Summary[1], 1); // EPE samples
        InterlockedAdd(Summary[4], uint(e.x));
        InterlockedAdd(Summary[5], uint(e.y));
        InterlockedMax(Summary[6], uint(e.x + e.y));
    } else {
        InterlockedAdd(Summary[7], 1); // synthetic occluder samples
    }
}
