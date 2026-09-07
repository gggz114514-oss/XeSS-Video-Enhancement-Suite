// Gate D4 diagnostic: OpenCV DIS weighted patch densification.
Texture2D<float> Current : register(t0);
Texture2D<float> Previous : register(t1);
Texture2D<float2> Sparse : register(t2);
RWTexture2D<float2> Dense : register(u0);
cbuffer Params : register(b0) { uint width; uint height; uint sparseWidth; uint sparseHeight; uint patch; uint stride; };

float bilinear(float x, float y)
{
    x = clamp(x, 0.0, (float)width - 1.001); y = clamp(y, 0.0, (float)height - 1.001);
    uint x0 = (uint)floor(x), y0 = (uint)floor(y);
    uint x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
    float fx = x - (float)x0, fy = y - (float)y0;
    float a = Previous.Load(int3(x0, y0, 0)), b = Previous.Load(int3(x1, y0, 0));
    float c = Previous.Load(int3(x0, y1, 0)), d = Previous.Load(int3(x1, y1, 0));
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    int y = (int)id.y, x = (int)id.x;
    int firstI = max(0, (int)ceil(((float)y - (float)(patch - 1)) / (float)stride));
    int lastI = min((int)sparseHeight - 1, y / (int)stride);
    int firstJ = max(0, (int)ceil(((float)x - (float)(patch - 1)) / (float)stride));
    int lastJ = min((int)sparseWidth - 1, x / (int)stride);
    float2 sumFlow = float2(0, 0); float sumWeight = 0;
    for (int si = firstI; si <= lastI; ++si) for (int sj = firstJ; sj <= lastJ; ++sj) {
        float2 flow = Sparse.Load(int3(sj, si, 0));
        float diff = bilinear((float)x + flow.x, (float)y + flow.y) - Current.Load(int3(x, y, 0));
        float weight = 1.0 / max(1.0, abs(diff));
        sumFlow += weight * flow; sumWeight += weight;
    }
    Dense[id.xy] = sumFlow / max(sumWeight, 1e-20);
}
