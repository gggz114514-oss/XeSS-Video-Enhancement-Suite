// fixed_block_v1 H1 output mapping.  The old product shader searched at
// floor(output * input/output) and scaled its integer displacement.  This
// pass preserves that exact coordinate mapping while moving the expensive
// search to input resolution.
Texture2D<float2> FlowIn : register(t0);
Texture2D<float2> FlowInAux : register(t1); // reserved by the shared root
RWTexture2D<float2> VelocityOut : register(u0);
RWTexture2D<float> ReservedOut : register(u1);

cbuffer Params : register(b0) {
    uint width; uint height; uint inputWidth; uint inputHeight;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    const int2 outP = int2(tid.xy);
    const int2 inP = min(
        int2(outP.x * inputWidth / width, outP.y * inputHeight / height),
        int2(inputWidth - 1, inputHeight - 1));
    const float2 scale = float2(width / float(inputWidth),
                                height / float(inputHeight));
    VelocityOut[outP] = FlowIn.Load(int3(inP, 0)) * scale;
}
