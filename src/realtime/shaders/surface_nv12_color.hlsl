Texture2D<float> LumaPlane : register(t0);
Texture2D<float2> ChromaPlane : register(t1);
RWTexture2D<float4> ColorOut : register(u0);
RWTexture2D<float> LumaOut : register(u1);

// constants.x/y = width/height, constants.z = matrix (0=bt601, 1=bt709),
// constants.w = range (0=limited, 1=full).
cbuffer Params : register(b0) {
    uint width;
    uint height;
    uint matrix_select;
    uint range_select;
};

// Bit-exact port of FFmpeg 7.1 libswscale x86 SSSE3 yuv_2_rgb.asm
// (yuv420_rgb24) for 8-bit limited/full range input. The CPU/FFmpeg
// reference pipeline decodes H.264 to yuv420p and converts with the same
// kernel, so matching it per byte keeps the CPU DIS and GPU Block XeSS
// inputs identical and the motion gates isolate motion only.
//
// The asm computes, per channel, in 13.3 fixed point:
//   Y8 = (Y << 3) - y_offset_lanes      (y_offset_lanes = 128 limited / 0 full)
//   U8 = (U << 3) - 1024 ; V8 = (V << 3) - 1024
//   C  = pmulhw(Y8, y_coeff) + pmulhw(V8, vr_coeff) + pmulhw(U8, ug/ub_coeff)
// where pmulhw(a,b) = (int16)((a * b) >> 16) (floor), coefficients are the
// 13-bit roundToInt16(coeff << 13) tables installed by sws_setColorspace
// Details from ff_yuv2rgb_coeffs:
//   bt601 {crv,cbu,cgu,cgv} = {104597,132201,25675,53279}
//   bt709 {crv,cbu,cgu,cgv} = {117489,138438,13975,34925}
// limited: cy = 65536*255/219, oy = 16<<16; full: cy = 65536, oy = 0 and
// chroma coefficients scaled by 224/255. Validated byte-exact against the
// reference decoder for the A (no VUI -> bt601) and B (VUI bt709) materials.
struct Coeffs {
    int y;    // luma coefficient (13-bit)
    int vr;   // V -> R
    int vg;   // V -> G (sign already applied)
    int ug;   // U -> G (sign already applied)
    int ub;   // U -> B
    int y_offset; // luma offset in <<3 domain (128*8 limited, 0 full)
};

static const Coeffs k601Limited = {9539, 13075, -6660, -3209, 16525, 128};
static const Coeffs k709Limited = {9539, 14686, -4366, -1747, 17305, 128};
// Full range keeps the asm coefficient layout (y_coeff = 8192, offsets 0);
// the 224/255-scaled chroma coefficients below reproduce the same tables.
// Full range is not used by the acceptance materials (all limited) and is
// calibrated to the documented tables, not byte-exact against this build.
static const Coeffs k601Full = {8192, 11485, -5850, -2819, 14516, 0};
static const Coeffs k709Full = {8192, 12901, -3835, -1535, 15201, 0};

int pmulhw(int a, int b) {
    // x86 pmulhw: high word of the 32-bit signed product (floor).
    // |a| <= 2040 and |b| <= 17305, so the product fits int32 with slack.
    return (a * b) >> 16;
}

int clip8(int v) { return clamp(v, 0, 255); }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(tid.xy);
    // R8/R8G8 UNORM loads decode value/255 exactly; *255 with rounding
    // recovers the raw 8-bit byte the CPU reference also starts from.
    int y = (int)(LumaPlane.Load(int3(p, 0)) * 255.0f + 0.5f);
    int2 uv = (int2)(ChromaPlane.Load(int3(p >> 1, 0)) * 255.0f + 0.5f);
    // swscale's unscaled 420 consumers share one chroma sample per 2x2
    // block (nearest), so chroma is intentionally not interpolated.
    bool use709 = matrix_select != 0u;
    bool full = range_select != 0u;
    // HLSL ternaries reject struct operands; select per component.
    Coeffs cf;
    cf.y = full ? (use709 ? k709Full.y : k601Full.y) : (use709 ? k709Limited.y : k601Limited.y);
    cf.vr = full ? (use709 ? k709Full.vr : k601Full.vr) : (use709 ? k709Limited.vr : k601Limited.vr);
    cf.vg = full ? (use709 ? k709Full.vg : k601Full.vg) : (use709 ? k709Limited.vg : k601Limited.vg);
    cf.ug = full ? (use709 ? k709Full.ug : k601Full.ug) : (use709 ? k709Limited.ug : k601Limited.ug);
    cf.ub = full ? (use709 ? k709Full.ub : k601Full.ub) : (use709 ? k709Limited.ub : k601Limited.ub);
    cf.y_offset = full ? 0 : 128;
    int y8 = (y << 3) - cf.y_offset;
    int u8 = (uv.x << 3) - 1024;
    int v8 = (uv.y << 3) - 1024;
    int yc = pmulhw(y8, cf.y);
    int r = clip8(yc + pmulhw(v8, cf.vr));
    int g = clip8(yc + pmulhw(u8, cf.ug) + pmulhw(v8, cf.vg));
    int b = clip8(yc + pmulhw(u8, cf.ub));
    float3 rgb = float3(r, g, b);
    ColorOut[p] = float4(rgb / 255.0f, 1.0f);
    // Motion luma stays consistent with the selected matrix family.
    float3 weights = use709 ? float3(0.2126f, 0.7152f, 0.0722f)
                            : float3(0.299f, 0.587f, 0.114f);
    LumaOut[p] = dot(rgb, weights) / 255.0f;
}
