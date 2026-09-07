// P5 final bridge: OpenCV DISOpticalFlow::calc ends with
//   resize(U[finest], fullSize, INTER_LINEAR); flow *= (1 << finest_scale);
// and the product XeSS contract needs velocity at OUTPUT resolution in
// output-resolution pixels per frame (xessSetVelocityScale(1,1), HIGH_RES_MV).
//
// mode 0 (GPU DIS source): sample the finest processed level (t0 = vr_work),
//   mapping output sample -> full-res coordinate -> level coordinate with the
//   exact half-pixel cv::resize convention, scaling by finestScale*outScale.
//   A companion dispatch with outScale=1 and the R32G32 diagnostic UAV bound
//   reproduces the OpenCV full-resolution flow bit-for-bit for acceptance.
// mode 1 (CPU reference endpoint): t1 = full-resolution current->previous flow
//   computed by cv2.DISOpticalFlow, uploaded once per frame for acceptance
//   only; velocity = bilinear(full flow) * outScale.
Texture2D<float2> LevelFlow : register(t0);
Texture2D<float2> FullFlow : register(t1);
RWTexture2D<float2> Velocity : register(u0);
cbuffer Params : register(b0)
{
    uint levelWidth;   // finest processed level width
    uint levelHeight;
    uint fullWidth;    // input (crop) width
    uint fullHeight;
    uint outWidth;     // XeSS output width
    uint outHeight;
    float finestScale; // 2^finest_scale (OpenCV final multiply)
    float outScale;    // output_width / input_width
    uint mode;         // 0 = level source (GPU DIS), 1 = full-res source (CPU ref)
    uint pad0;
};

float2 bilerp(Texture2D<float2> tex, float2 p, uint2 dims)
{
    p = clamp(p, float2(0.0, 0.0), float2(dims) - 1.0);
    uint2 lo = uint2(floor(p));
    uint2 hi = min(lo + 1, dims - 1);
    float2 f = p - float2(lo);
    float2 a = tex.Load(int3(lo, 0));
    float2 b = tex.Load(int3(uint2(hi.x, lo.y), 0));
    float2 c = tex.Load(int3(uint2(lo.x, hi.y), 0));
    float2 d = tex.Load(int3(hi, 0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= outWidth || id.y >= outHeight) return;
    float2 value;
    if (mode == 0) {
        // OpenCV full-res coordinate of this output sample ...
        float2 full = (float2(id.xy) + 0.5) / outScale - 0.5;
        // ... composed with the OpenCV final resize mapping into the level grid.
        float2 p = (full + 0.5) * float2(levelWidth, levelHeight) / float2(fullWidth, fullHeight) - 0.5;
        value = bilerp(LevelFlow, p, uint2(levelWidth, levelHeight)) * (finestScale * outScale);
    } else {
        float2 p = (float2(id.xy) + 0.5) / outScale - 0.5;
        value = bilerp(FullFlow, p, uint2(fullWidth, fullHeight)) * outScale;
    }
    Velocity[id.xy] = value;
}
