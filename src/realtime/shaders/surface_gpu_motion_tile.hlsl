// H2 fixed_block_v1 tile organization.  This is still the same 17x17
// integer candidate search and 3x3 SAD patch as fixed_block_v1_h1, but the
// 8x8 workgroup cooperatively loads a 26x26 luma slab (8 + 2*8 + 2*1) for
// each source image.  Neighboring threads therefore reuse global loads from
// the same candidate/search neighborhood.
//
// Confidence: after the search the shader re-evaluates the four axis
// neighbours of the winning delta.  The margin between the winner and the
// cheapest neighbour is a local-minimum sharpness measure and is published
// as conf = (second - best) / (second + best + eps) when constants[4] is
// set.  The repair pass and the responsive-mask pass use this value to
// mark unreliable tiles instead of trusting flat-region ties.
Texture2D<float> PreviousLuma : register(t0);
Texture2D<float> CurrentLuma : register(t1);
RWTexture2D<float2> FlowOut : register(u0);
RWTexture2D<float> ConfidenceOut : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
    uint write_confidence; float center_bias; uint search_radius; uint unused3;
};

static const int SEARCH_RADIUS = 8;
static const int PATCH_RADIUS = 1;
static const int WG_X = 8;
static const int WG_Y = 8;
static const int TILE_W = WG_X + 2 * SEARCH_RADIUS + 2 * PATCH_RADIUS;
static const int TILE_H = WG_Y + 2 * SEARCH_RADIUS + 2 * PATCH_RADIUS;

groupshared float PreviousTile[TILE_H][TILE_W];
groupshared float CurrentTile[TILE_H][TILE_W];

int2 cp(int2 p) {
    return clamp(p, int2(0, 0),
                 int2(int(inputWidth) - 1, int(inputHeight) - 1));
}

float patch_cost_tile(int2 p, int2 q, int2 tile_origin) {
    float cost = 0.0f;
    [unroll] for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; ++dy)
        [unroll] for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; ++dx) {
            const int2 ap = cp(p + int2(dx, dy)) - tile_origin;
            const int2 bp = cp(q + int2(dx, dy)) - tile_origin;
            cost += abs(CurrentTile[ap.y][ap.x] - PreviousTile[bp.y][bp.x]);
        }
    return cost;
}

float2 search_one_tile(int2 p, int2 tile_origin, out float confidence) {
    float best = 1e20f;
    float best_raw = 1e20f;
    int2 best_delta = 0;
    const int radius = clamp(int(search_radius), 1, SEARCH_RADIUS);
    [loop] for (int dy = -radius; dy <= radius; ++dy)
        [loop] for (int dx = -radius; dx <= radius; ++dx) {
            const int2 delta = int2(dx, dy);
            // Optional center bias (Tikhonov-style pull toward zero).  With
            // it, flat-region ties break coherently toward the smallest
            // displacement in BOTH directions, which is what makes the raw
            // round trip self-consistent; without it the winner inside a
            // tie is arbitrary and the forward/backward pair disagrees.
            const float raw = patch_cost_tile(p, cp(p + delta), tile_origin);
            const float cost = raw + center_bias * (abs(dx) + abs(dy));
            if (cost < best) {
                best = cost;
                best_raw = raw;
                best_delta = delta;
            }
        }
    // Local-minimum sharpness: the cheapest cost among the four axis
    // neighbours of the winner.  A winner surrounded by equally cheap
    // candidates is a tie inside a flat or repetitive region and must not
    // be trusted by downstream repair.
    float second = 1e20f;
    [unroll] for (int k = 0; k < 4; ++k) {
        const int2 step = k == 0 ? int2(1, 0) : k == 1 ? int2(-1, 0)
                        : k == 2 ? int2(0, 1) : int2(0, -1);
        const int2 q = cp(p + best_delta + step);
        second = min(second, patch_cost_tile(p, q, tile_origin));
    }
    confidence = saturate((second - best_raw) / (second + best_raw + 1e-3f));
    return float2(best_delta);
}

[numthreads(WG_X, WG_Y, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          uint3 gid : SV_GroupID,
          uint3 ltid : SV_GroupThreadID) {
    const int2 group_base = int2(gid.xy) * int2(WG_X, WG_Y);
    const int2 tile_origin = group_base - int2(SEARCH_RADIUS + PATCH_RADIUS,
                                               SEARCH_RADIUS + PATCH_RADIUS);
    const uint local_linear = ltid.y * WG_X + ltid.x;
    const uint tile_count = TILE_W * TILE_H;
    // All lanes participate in the barrier, including the partial workgroup
    // at an odd-sized image edge.
    for (uint i = local_linear; i < tile_count; i += WG_X * WG_Y) {
        const int tx = int(i % TILE_W);
        const int ty = int(i / TILE_W);
        const int2 source = cp(tile_origin + int2(tx, ty));
        CurrentTile[ty][tx] = CurrentLuma.Load(int3(source, 0));
        PreviousTile[ty][tx] = PreviousLuma.Load(int3(source, 0));
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid.x >= inputWidth || tid.y >= inputHeight) return;
    const int2 p = int2(tid.xy);
    float confidence;
    FlowOut[p] = search_one_tile(p, tile_origin, confidence);
    if (write_confidence) ConfidenceOut[p] = confidence;
}
