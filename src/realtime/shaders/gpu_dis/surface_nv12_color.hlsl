Texture2D<float> LumaPlane : register(t0);
Texture2D<float2> ChromaPlane : register(t1);
RWTexture2D<float4> ColorOut : register(u0);
RWTexture2D<float> LumaOut : register(u1);

// constants.x/y = width/height, constants.z = matrix (0=bt601, 1=bt709),
// constants.w = range (0=limited, 1=full). Keep this conversion identical to
// the full-GPU Block path; DIS changes only motion/mask, not the XeFG color
// contract.
cbuffer Params : register(b0) {
    uint width;
    uint height;
    uint matrix_select;
    uint range_select;
};

struct Coeffs {
    int y;
    int vr;
    int vg;
    int ug;
    int ub;
    int y_offset;
};

static const Coeffs k601Limited = {9539, 13075, -6660, -3209, 16525, 128};
static const Coeffs k709Limited = {9539, 14686, -4366, -1747, 17305, 128};
static const Coeffs k601Full = {8192, 11485, -5850, -2819, 14516, 0};
static const Coeffs k709Full = {8192, 12901, -3835, -1535, 15201, 0};

int pmulhw(int a, int b) { return (a * b) >> 16; }
int clip8(int v) { return clamp(v, 0, 255); }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    int2 p = int2(tid.xy);
    float y = LumaPlane.Load(int3(p, 0));
    float2 uv = ChromaPlane.Load(int3(p / 2, 0));
    int yi = (int)(LumaPlane.Load(int3(p, 0)) * 255.0f + 0.5f);
    int2 uvi = (int2)(ChromaPlane.Load(int3(p / 2, 0)) * 255.0f + 0.5f);
    bool use709 = matrix_select != 0u;
    bool full = range_select != 0u;
    Coeffs cf;
    cf.y = full ? (use709 ? k709Full.y : k601Full.y) : (use709 ? k709Limited.y : k601Limited.y);
    cf.vr = full ? (use709 ? k709Full.vr : k601Full.vr) : (use709 ? k709Limited.vr : k601Limited.vr);
    cf.vg = full ? (use709 ? k709Full.vg : k601Full.vg) : (use709 ? k709Limited.vg : k601Limited.vg);
    cf.ug = full ? (use709 ? k709Full.ug : k601Full.ug) : (use709 ? k709Limited.ug : k601Limited.ug);
    cf.ub = full ? (use709 ? k709Full.ub : k601Full.ub) : (use709 ? k709Limited.ub : k601Limited.ub);
    cf.y_offset = full ? 0 : 128;
    int y8 = (yi << 3) - cf.y_offset;
    int u8 = (uvi.x << 3) - 1024;
    int v8 = (uvi.y << 3) - 1024;
    int yc = pmulhw(y8, cf.y);
    int r = clip8(yc + pmulhw(v8, cf.vr));
    int g = clip8(yc + pmulhw(u8, cf.ug) + pmulhw(v8, cf.vg));
    int b = clip8(yc + pmulhw(u8, cf.ub));
    float3 rgb = float3(r, g, b);
    ColorOut[p] = float4(rgb / 255.0f, 1.0f);
    float3 weights = use709 ? float3(0.2126f, 0.7152f, 0.0722f)
                            : float3(0.299f, 0.587f, 0.114f);
    LumaOut[p] = dot(rgb, weights) / 255.0f;
}
