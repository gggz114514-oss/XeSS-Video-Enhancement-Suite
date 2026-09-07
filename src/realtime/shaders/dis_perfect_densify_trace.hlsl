// Diagnostic companion to dis_perfect_densify.hlsl.
// The dense result uses the production arithmetic; when (traceX,traceY) is
// reached, up to four overlap contributions are written to Trace:
// rows 0..3: flow.x, flow.y, photometric diff, weight
// rows 4..7: xm, ym, lo.x, lo.y (row 4 is overwritten with a,b,c,d below)
// rows 8..11: w1, w2, v1, v2
// rows 12..15: interpolated value, sumFlow.x, sumFlow.y, sumWeight after add
// Trace[0,16] is the final x/y, sumWeight, contribution count; [0,17] is the
// staged base/count; [0,18] is the selected overlap window.
Texture2D<float> Current : register(t0);
Texture2D<float> Previous : register(t1);
Texture2D<float2> Sparse : register(t2);
RWTexture2D<float2> Dense : register(u0);
RWTexture2D<float4> Trace : register(u1);
cbuffer Params : register(b0) { uint width; uint height; uint sparseWidth; uint sparseHeight; uint patch; uint stride; uint traceX; uint traceY; };

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
    float a = Previous.Load(int3(lo, 0));
    float b = Previous.Load(int3(uint2(hi.x, lo.y), 0));
    float c = Previous.Load(int3(uint2(lo.x, hi.y), 0));
    float d = Previous.Load(int3(hi, 0));
    precise float t = (x - (float)lo.x) * (y - (float)lo.y) * d;
    t = t + ((float)hi.x - x) * (y - (float)lo.y) * c;
    t = t + (x - (float)lo.x) * ((float)hi.y - y) * b;
    t = t + ((float)hi.x - x) * ((float)hi.y - y) * a;
    return t;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint lane = tid.y * 8 + tid.x;
    uint tileX = gid.x * 8, tileY = gid.y * 8;
    if (lane == 0)
    {
        int bx = max(0, (int)tileX - (int)patch + 1);
        int by = max(0, (int)tileY - (int)patch + 1);
        gBaseX = (uint)((bx + (int)stride - 1) / (int)stride);
        gBaseY = (uint)((by + (int)stride - 1) / (int)stride);
        gBaseX = min(gBaseX, sparseWidth - 1);
        gBaseY = min(gBaseY, sparseHeight - 1);
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
    uint lastX = min(gBaseX + gCountX - 1, x / stride);
    uint lastY = min(gBaseY + gCountY - 1, y / stride);
    uint firstX = (x + 1 <= patch) ? 0 : (x - patch + stride) / stride;
    uint firstY = (y + 1 <= patch) ? 0 : (y - patch + stride) / stride;
    firstX = min(firstX, lastX); firstY = min(firstY, lastY);
    float2 sumFlow = 0.0; float sumWeight = 0.0; uint k = 0;
    bool selected = x == traceX && y == traceY;
    if (selected) {
        Trace[uint2(0, 17)] = float4(gBaseX, gBaseY, gCountX, gCountY);
        Trace[uint2(0, 18)] = float4(firstX, firstY, lastX, lastY);
    }
    for (uint sy = firstY; sy <= lastY; ++sy)
        for (uint sx = firstX; sx <= lastX; ++sx)
        {
            float2 flow = gSparse[sy - gBaseY][sx - gBaseX];
            float2 coord = float2((float)x + flow.x, (float)y + flow.y);
            float xm = clamp(coord.x, 0.0, (float)width - 1.001);
            float ym = clamp(coord.y, 0.0, (float)height - 1.001);
            uint2 lo = uint2(floor(float2(xm, ym)));
            uint2 hi = min(lo + 1, uint2(width - 1, height - 1));
            float w1 = xm - (float)lo.x, w2 = (float)hi.x - xm;
            float v1 = ym - (float)lo.y, v2 = (float)hi.y - ym;
            float ta = Previous.Load(int3(lo, 0));
            float tb = Previous.Load(int3(uint2(hi.x, lo.y), 0));
            float tc = Previous.Load(int3(uint2(lo.x, hi.y), 0));
            float td = Previous.Load(int3(hi, 0));
            float t = bilinear_previous(coord.x, coord.y);
            float diff = t - Current.Load(int3(x, y, 0));
            float weight = 1.0 / max(1.0, abs(diff));
            sumFlow += weight * flow; sumWeight += weight;
            if (selected && k < 4) {
                Trace[uint2(0, k)] = float4(flow.x, flow.y, diff, weight);
                Trace[uint2(1, k)] = float4(xm, ym, lo.x, lo.y);
                Trace[uint2(2, k)] = float4(w1, w2, v1, v2);
                Trace[uint2(3, k)] = float4(t, sumFlow.x, sumFlow.y, sumWeight);
                Trace[uint2(1, 4 + k)] = float4(ta, tb, tc, td);
            }
            ++k;
        }
    Dense[uint2(x, y)] = sumFlow / max(sumWeight, 1e-20);
    if (selected) Trace[uint2(0, 16)] = float4(Dense[uint2(x, y)].x, Dense[uint2(x, y)].y, sumWeight, k);
}
