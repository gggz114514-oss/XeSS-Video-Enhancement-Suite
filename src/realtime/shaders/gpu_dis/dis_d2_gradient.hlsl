// Gate D2 diagnostic only. This is OpenCV spatialGradient's 3x3 Sobel
// derivative, with the documented zero border and signed 32-bit output. The
// production port may pack to R16G16_SINT only after a precision gate.
Texture2D<float> Source : register(t0);
RWTexture2D<int2> Destination : register(u0);
cbuffer Params : register(b0) { uint srcWidth; uint srcHeight; uint dstWidth; uint dstHeight; };

int reflect101(int p, int extent)
{
    if (extent <= 1) return 0;
    if (p < 0) return -p;
    if (p >= extent) return 2 * extent - p - 2;
    return p;
}

float sample_reflect101(int x, int y)
{
    return Source.Load(int3(reflect101(x, int(srcWidth)), reflect101(y, int(srcHeight)), 0));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dstWidth || id.y >= dstHeight) return;
    int x = int(id.x), y = int(id.y);
    float gx = -sample_reflect101(x - 1, y - 1) - 2.0 * sample_reflect101(x - 1, y) - sample_reflect101(x - 1, y + 1)
             + sample_reflect101(x + 1, y - 1) + 2.0 * sample_reflect101(x + 1, y) + sample_reflect101(x + 1, y + 1);
    float gy = -sample_reflect101(x - 1, y - 1) - 2.0 * sample_reflect101(x, y - 1) - sample_reflect101(x + 1, y - 1)
             + sample_reflect101(x - 1, y + 1) + 2.0 * sample_reflect101(x, y + 1) + sample_reflect101(x + 1, y + 1);
    // spatialGradient uses a reflected border in the perpendicular axis,
    // while the derivative at the edge along its own axis is zero.
    if (x == 0 || x + 1 >= srcWidth) gx = 0.0;
    if (y == 0 || y + 1 >= srcHeight) gy = 0.0;
    Destination[id.xy] = int2((int)round(gx), (int)round(gy));
}
