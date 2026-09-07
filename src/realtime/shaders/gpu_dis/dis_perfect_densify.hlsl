// GPU DIS perfect-port P3: tiled OpenCV weighted patch densification.
// One 8x8 output tile is a workgroup.  The sparse vectors covering the tile's
// expanded 8-pixel halo are staged in groupshared memory; every lane then
// performs the exact bounded overlap/photometric weighting for one output
// pixel.  No overlap contribution or 1/max(1,abs(diff)) weight is dropped.
Texture2D<float> Current : register(t0);
Texture2D<float> Previous : register(t1);
Texture2D<float2> Sparse : register(t2);
RWTexture2D<float2> Dense : register(u0);
cbuffer Params : register(b0) { uint width; uint height; uint sparseWidth; uint sparseHeight; uint patch; uint stride; };

groupshared float2 gSparse[6][6];
groupshared uint gBaseX;
groupshared uint gBaseY;
groupshared uint gCountX;
groupshared uint gCountY;

float bilinear_previous(float x, float y)
{
    x = clamp(x, 0.0, (float)width - 1.001);
    y = clamp(y, 0.0, (float)height - 1.001);
    uint2 lo = uint2(floor(float2(x, y)));
    uint2 hi = min(lo + 1, uint2(width - 1, height - 1));
    float2 f = float2(x, y) - float2(lo);
    float a = Previous.Load(int3(lo, 0));
    float b = Previous.Load(int3(uint2(hi.x, lo.y), 0));
    float c = Previous.Load(int3(uint2(lo.x, hi.y), 0));
    float d = Previous.Load(int3(hi, 0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint lane = tid.y * 8 + tid.x;
    uint tileX = gid.x * 8, tileY = gid.y * 8;
    if (lane == 0)
    {
        // Sparse patches whose [js*stride, js*stride+patch-1] interval can
        // overlap any pixel in this tile have top-left coordinates in the
        // expanded tile interval [tile- patch+1, tile+7].
        int bx = max(0, (int)tileX - (int)patch + 1);
        int by = max(0, (int)tileY - (int)patch + 1);
        gBaseX = (uint)((bx + (int)stride - 1) / (int)stride);
        gBaseY = (uint)((by + (int)stride - 1) / (int)stride);
        uint ex = min(sparseWidth - 1, (tileX + 7) / stride);
        uint ey = min(sparseHeight - 1, (tileY + 7) / stride);
        gCountX = ex >= gBaseX ? ex - gBaseX + 1 : 0;
        gCountY = ey >= gBaseY ? ey - gBaseY + 1 : 0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint count = gCountX * gCountY;
    if (lane < count && lane < 36)
    {
        uint localY = lane / gCountX, localX = lane - localY * gCountX;
        gSparse[localY][localX] = Sparse.Load(int3(gBaseX + localX, gBaseY + localY, 0)).xy;
    }
    GroupMemoryBarrierWithGroupSync();
    uint x = tileX + tid.x, y = tileY + tid.y;
    if (x >= width || y >= height || gCountX == 0 || gCountY == 0) return;
    uint firstX = (x + 1 <= patch) ? gBaseX : max(gBaseX, (x - patch + stride) / stride);
    uint firstY = (y + 1 <= patch) ? gBaseY : max(gBaseY, (y - patch + stride) / stride);
    uint lastX = min(gBaseX + gCountX - 1, x / stride);
    uint lastY = min(gBaseY + gCountY - 1, y / stride);
    float2 sumFlow = 0.0;
    float sumWeight = 0.0;
    for (uint sy = firstY; sy <= lastY; ++sy)
        for (uint sx = firstX; sx <= lastX; ++sx)
        {
            float2 flow = gSparse[sy - gBaseY][sx - gBaseX];
            float diff = bilinear_previous((float)x + flow.x, (float)y + flow.y) - Current.Load(int3(x, y, 0));
            float weight = 1.0 / max(1.0, abs(diff));
            sumFlow += weight * flow;
            sumWeight += weight;
        }
    Dense[uint2(x, y)] = sumFlow / max(sumWeight, 1e-20);
}
