// P4 standalone variant of VariationalRefinement's first derivative pass.
// The harness uses an R8_UNORM diagnostic surface, while OpenCV promotes the
// CV_8U samples to their 0..255 float values before remap/Sobel. Keep the
// production R32F shader unchanged and expand the diagnostic SRV here. The
// reference uses Sobel(..., ksize=1) == the plain central difference and a
// CV_32F remap with unquantized bilinear fractions.
Texture2D<float> Current : register(t0);
Texture2D<float> Previous : register(t1);
Texture2D<float2> Flow : register(t2);
RWTexture2D<float> Ix : register(u0);
RWTexture2D<float> Iy : register(u1);
RWTexture2D<float> Iz : register(u2);
RWTexture2D<float> Ixz : register(u3);
RWTexture2D<float> Iyz : register(u4);
cbuffer Params : register(b0) { uint width; uint height; };

int2 clamp_xy(int2 p) { return clamp(p, int2(0, 0), int2(width - 1, height - 1)); }
float cur_at(int2 p) { p = clamp_xy(p); return Current.Load(int3(p, 0)) * 255.0; }
float2 flow_at(int2 p) { p = clamp_xy(p); return Flow.Load(int3(p, 0)); }
float prev_linear(float2 p)
{
    p = clamp(p, 0.0, float2(width - 1, height - 1));
    int2 q = int2(floor(p)); int2 r = min(q + 1, int2(width - 1, height - 1));
    float2 f = p - q;
    float a = Previous.Load(int3(q, 0)), b = Previous.Load(int3(int2(r.x, q.y), 0));
    float c = Previous.Load(int3(int2(q.x, r.y), 0)), d = Previous.Load(int3(r, 0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y) * 255.0;
}
float warped_at(int2 p)
{
    p = clamp_xy(p); float2 f = flow_at(p); return prev_linear(float2(p) + f);
}
float avg_at(int2 p) { p = clamp_xy(p); return 0.5 * (cur_at(p) + warped_at(p)); }
float iz_at(int2 p) { p = clamp_xy(p); return warped_at(p) - cur_at(p); }
float diff_x(float at[3][3]) { return at[1][2] - at[1][0]; }
float diff_y(float at[3][3]) { return at[2][1] - at[0][1]; }
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    int2 p = int2(id.xy); float a[3][3]; float z[3][3];
    [unroll] for (int yy = 0; yy < 3; ++yy) [unroll] for (int xx = 0; xx < 3; ++xx) {
        int2 q = p + int2(xx - 1, yy - 1); a[yy][xx] = avg_at(q); z[yy][xx] = iz_at(q);
    }
    Ix[p] = diff_x(a); Iy[p] = diff_y(a); Iz[p] = z[1][1];
    Ixz[p] = diff_x(z); Iyz[p] = diff_y(z);
}
