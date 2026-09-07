// OpenCV 5.0.0 VariationalRefinement second derivative pass. The reference
// calls Sobel(..., ksize=1, BORDER_REPLICATE) on the first-order maps, so
// every second derivative is the plain central difference [-1, 0, 1]:
// Ixx = dIx/dx, Ixy = dIx/dy, Iyy = dIy/dy.
Texture2D<float> Ix : register(t0);
Texture2D<float> Iy : register(t1);
RWTexture2D<float> Ixx : register(u0);
RWTexture2D<float> Ixy : register(u1);
RWTexture2D<float> Iyy : register(u2);
cbuffer Params : register(b0) { uint width; uint height; };
int2 clamp_xy(int2 p) { return clamp(p, int2(0, 0), int2(width - 1, height - 1)); }
float load_x(int2 p) { return Ix.Load(int3(clamp_xy(p), 0)); }
float load_y(int2 p) { return Iy.Load(int3(clamp_xy(p), 0)); }
float diff_x_x(int2 p) { return load_x(p + int2(1, 0)) - load_x(p - int2(1, 0)); }
float diff_y_x(int2 p) { return load_x(p + int2(0, 1)) - load_x(p - int2(0, 1)); }
float diff_y_y(int2 p) { return load_y(p + int2(0, 1)) - load_y(p - int2(0, 1)); }
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    int2 p = int2(id.xy); Ixx[p] = diff_x_x(p); Ixy[p] = diff_y_x(p); Iyy[p] = diff_y_y(p);
}
