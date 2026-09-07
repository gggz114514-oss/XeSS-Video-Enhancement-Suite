// Motion repair pass (H2 confidence follow-up).
//
// Runs once after both directional searches when --motion-repair selects a
// mode.  Inputs: the raw forward (current->previous) and backward
// (previous->current) fixed-block fields, the forward confidence field from
// the search and both luma planes.  Output: a repaired forward field that
// the velocity mapping, the responsive-mask pass, the XeSS execute and the
// flow diagnostics all consume.  The raw backward field is left untouched.
//
// mode 1 - propagate (repair A): pixels whose forward/backward round trip
//   disagrees (occlusion or bad tie) or whose search confidence is low get
//   the component-wise median of trusted 5x5 neighbours.  Propagation does
//   not cross strong luma edges (edge-aware); with fewer than three trusted
//   neighbours the vector falls back to zero (global-or-zero).
//
// mode 2 - refine (repair B): only outliers get a cheap radius-2 local
//   re-search around the raw winner.  The joint cost adds the round-trip
//   disagreement |d + backward(p + d)| so the refined vector must both fit
//   photometrically and agree with the opposing field.
//
// mode 0 (off) keeps the raw field; the probe skips this dispatch and the
// ablation stays possible without rebuilding.
Texture2D<float2> ForwardFlowIn : register(t0);
Texture2D<float2> BackwardFlowIn : register(t1);
Texture2D<float> ConfidenceIn : register(t2);
Texture2D<float> CurrentLuma : register(t3);
Texture2D<float> PreviousLuma : register(t4);
RWTexture2D<float2> ForwardFlowOut : register(u0);
RWTexture2D<float> ReservedOut : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint unused0; uint unused1;
    uint repair_mode; uint unused2; uint unused3; uint unused4;
};

int2 cp(int2 p) {
    return clamp(p, int2(0, 0), int2(int(width) - 1, int(height) - 1));
}

float consistency_at(int2 p) {
    float2 f = ForwardFlowIn.Load(int3(p, 0));
    int2 q = cp(p + int2(round(f)));
    float2 b = BackwardFlowIn.Load(int3(q, 0));
    return length(f + b);
}

float patch_sad(int2 p, int2 q) {
    float cost = 0.0f;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
        [unroll] for (int dx = -1; dx <= 1; ++dx) {
            cost += abs(CurrentLuma.Load(int3(cp(p + int2(dx, dy)), 0)) -
                        PreviousLuma.Load(int3(cp(q + int2(dx, dy)), 0)));
        }
    return cost;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    const int2 p = int2(tid.xy);
    const float2 f = ForwardFlowIn.Load(int3(p, 0));
    const float confidence = ConfidenceIn.Load(int3(p, 0));
    const float consistency = consistency_at(p);

    bool outlier = consistency > 2.0f || confidence < 0.15f;
    float2 repaired = f;

    if (repair_mode == 1u) {
        if (outlier) {
            // Component-wise median of trusted, edge-compatible neighbours.
            float xs[25], ys[25];
            int count = 0;
            [loop] for (int dy = -2; dy <= 2; ++dy)
            [loop] for (int dx = -2; dx <= 2; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const int2 n = cp(p + int2(dx, dy));
                if (ConfidenceIn.Load(int3(n, 0)) < 0.15f) continue;
                if (consistency_at(n) > 2.0f) continue;
                // Edge-aware: do not propagate across a strong luma edge.
                if (abs(CurrentLuma.Load(int3(n, 0)) -
                        CurrentLuma.Load(int3(p, 0))) > 0.1f) continue;
                float2 nf = ForwardFlowIn.Load(int3(n, 0));
                xs[count] = nf.x; ys[count] = nf.y; ++count;
            }
            if (count >= 3) {
                // Independent per-component insertion sorts (n <= 24); the
                // median of each component is taken separately.
                [loop] for (int i = 1; i < count; ++i) {
                    float vx = xs[i];
                    int j = i - 1;
                    [loop] while (j >= 0 && xs[j] > vx) { xs[j + 1] = xs[j]; --j; }
                    xs[j + 1] = vx;
                }
                [loop] for (int i = 1; i < count; ++i) {
                    float vy = ys[i];
                    int j = i - 1;
                    [loop] while (j >= 0 && ys[j] > vy) { ys[j + 1] = ys[j]; --j; }
                    ys[j + 1] = vy;
                }
                repaired = float2(xs[count / 2], ys[count / 2]);
            } else {
                // global-or-zero fallback: no trustworthy support at all.
                repaired = float2(0.0f, 0.0f);
            }
        }
    } else if (repair_mode == 2u) {
        if (outlier) {
            // Radius-2 joint re-search around the raw winner.  The round
            // trip term is weighted against the 3x3 SAD of the raw winner
            // so photometric fit and consistency trade off in the same unit
            // domain (SAD is on normalized luma).
            const float base_sad = patch_sad(p, cp(p + int2(round(f))));
            const float w = 0.05f * max(base_sad, 0.05f);
            float best_cost = 1e20f;
            int2 best_delta = int2(round(f));
            [loop] for (int dy = -2; dy <= 2; ++dy)
            [loop] for (int dx = -2; dx <= 2; ++dx) {
                const int2 d = int2(round(f)) + int2(dx, dy);
                const int2 q = cp(p + d);
                float2 b = BackwardFlowIn.Load(int3(q, 0));
                const float cost = patch_sad(p, q) + w * length(float2(d) + b);
                if (cost < best_cost) { best_cost = cost; best_delta = d; }
            }
            repaired = float2(best_delta);
        }
    }
    ForwardFlowOut[p] = repaired;
}
