// OpenCV 5.0.0 VariationalRefinement data + smoothness system assembly.
// One invocation owns one pixel, so edge terms are accumulated without the
// write races of a scatter implementation. The equations are the CPU
// ComputeDataTerm + horizontal/vertical smoothness passes, with the same
// alpha/delta/gamma/epsilon contract used by DIS Fast.
Texture2D<float> Ix : register(t0);
Texture2D<float> Iy : register(t1);
Texture2D<float> Iz : register(t2);
Texture2D<float> Ixx : register(t3);
Texture2D<float> Ixy : register(t4);
Texture2D<float> Iyy : register(t5);
Texture2D<float> Ixz : register(t6);
Texture2D<float> Iyz : register(t7);
Texture2D<float2> DFlow : register(t8);
Texture2D<float2> BaseFlow : register(t9);
Texture2D<float2> WorkFlow : register(t10);
RWTexture2D<float> A11 : register(u0);
RWTexture2D<float> A12 : register(u1);
RWTexture2D<float> A22 : register(u2);
RWTexture2D<float> B1 : register(u3);
RWTexture2D<float> B2 : register(u4);
RWTexture2D<float> Weights : register(u5);
cbuffer Params : register(b0) { uint width; uint height; float alpha; float delta; float gamma; float epsilon; };
#ifdef DIS_VR_EXACT
#include "dis_cpu_rounding.hlsli"
#else
float cpu_divide(float a, float b) { return a / b; }
float cpu_sqrt(float a) { return sqrt(a); }
#endif

int2 clamp_xy(int2 p) { return clamp(p, int2(0, 0), int2(width - 1, height - 1)); }
float2 flow_at(Texture2D<float2> f, int2 p) { return f.Load(int3(clamp_xy(p), 0)); }
// OpenCV's horizontal pass stores one forward-gradient weight per pixel and
// reuses it for both the right and down edges. Left/up edges use the source
// pixel's corresponding stored weight. The sum keeps OpenCV's sequential
// association: ux*ux + vx*vx + uy*uy + vy*vy + epsilon^2.
float forward_weight(float2 work, float2 right_work, float2 down_work)
{
    float ux = right_work.x - work.x, vx = right_work.y - work.y;
    float uy = down_work.x - work.x, vy = down_work.y - work.y;
    return cpu_divide(alpha * 0.5, cpu_sqrt(ux * ux + vx * vx + uy * uy + vy * vy + epsilon * epsilon));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    int2 p = int2(id.xy); float2 dw = DFlow.Load(int3(p, 0));
    float ix = Ix.Load(int3(p,0)), iy = Iy.Load(int3(p,0)), iz = Iz.Load(int3(p,0));
    float zeta2 = 0.010000000707805157f, eps2 = epsilon * epsilon;
    float deriv = ix * ix + iy * iy + zeta2;
    float residual = iz + ix * dw.x + iy * dw.y;
    float w = cpu_divide(cpu_divide(delta * 0.5, cpu_sqrt(cpu_divide(residual * residual, deriv) + eps2)), deriv);
    // Parenthesization follows OpenCV exactly: weight * (Ix*Ix) + zeta^2.
    float a11 = w * (ix * ix) + zeta2, a12 = w * (ix * iy);
    float a22 = w * (iy * iy) + zeta2;
    float b1 = -(w * (iz * ix)), b2 = -(w * (iz * iy));

    float ixx = Ixx.Load(int3(p,0)), ixy = Ixy.Load(int3(p,0)), iyy = Iyy.Load(int3(p,0));
    float ixz = Ixz.Load(int3(p,0)), iyz = Iyz.Load(int3(p,0));
    float n1 = ixx * ixx + ixy * ixy + zeta2, n2 = iyy * iyy + ixy * ixy + zeta2;
    float qx = ixz + ixx * dw.x + ixy * dw.y, qy = iyz + ixy * dw.x + iyy * dw.y;
    float gw = cpu_divide(gamma * 0.5, cpu_sqrt(cpu_divide(qx * qx, n1) + cpu_divide(qy * qy, n2) + eps2));
    a11 += gw * (cpu_divide(ixx * ixx, n1) + cpu_divide(ixy * ixy, n2));
    a12 += gw * (cpu_divide(ixx * ixy, n1) + cpu_divide(ixy * iyy, n2));
    a22 += gw * (cpu_divide(ixy * ixy, n1) + cpu_divide(iyy * iyy, n2));
    b1 -= gw * (cpu_divide(ixx * ixz, n1) + cpu_divide(ixy * iyz, n2));
    b2 -= gw * (cpu_divide(ixy * ixz, n1) + cpu_divide(iyy * iyz, n2));

    float2 work = WorkFlow.Load(int3(p,0)), base = BaseFlow.Load(int3(p,0));
    // OpenCV's implementation stores one horizontal weight per pixel and
    // deliberately reuses that array for the vertical pass.
    float2 right_work = flow_at(WorkFlow,p+int2(1,0));
    float2 down_work = flow_at(WorkFlow,p+int2(0,1));
    float wcur = forward_weight(work, right_work, down_work);
    float edge;
    // OpenCV HorPass edge (p, p+right): b1[p] += w*(W[right]-W[p]) and the
    // symmetric b1[right] -= w*(W[right]-W[p]); the receiving pixel therefore
    // accumulates the NEGATED difference for its left/up edges. The original
    // port added the negated form with a positive sign, flipping the
    // smoothness contribution of every left/up edge.
    // CPU scatters red edges before black edges. Thus black destinations
    // receive left/up contributions before their own right/down contributions.
    // A gather may reorder ownership, but must retain that FP32 association.
    uint colour = 0;
#ifdef DIS_VR_EXACT
    colour = (id.x + id.y) & 1u;
#endif
    [unroll] for (uint pass = 0; pass < 2; ++pass) {
        bool forward = ((pass + colour) & 1u) == 0;
        if (forward && p.x + 1 < (int)width) {
            edge = wcur; a11 += edge; a22 += edge;
            float2 d = edge * (flow_at(BaseFlow,p+int2(1,0)) - base); b1 += d.x; b2 += d.y;
        } else if (!forward && p.x > 0) {
            int2 left_p = p - int2(1, 0);
            edge = forward_weight(flow_at(WorkFlow,left_p), work, flow_at(WorkFlow,left_p+int2(0,1)));
            a11 += edge; a22 += edge;
            float2 d = edge * (base - flow_at(BaseFlow,left_p)); b1 -= d.x; b2 -= d.y;
        }
    }
    [unroll] for (uint pass = 0; pass < 2; ++pass) {
        bool forward = ((pass + colour) & 1u) == 0;
        if (forward && p.y + 1 < (int)height) {
            edge = wcur; a11 += edge; a22 += edge;
            float2 d = edge * (flow_at(BaseFlow,p+int2(0,1)) - base); b1 += d.x; b2 += d.y;
        } else if (!forward && p.y > 0) {
            int2 up = p - int2(0, 1);
            edge = forward_weight(flow_at(WorkFlow,up), flow_at(WorkFlow,up+int2(1,0)), work);
            a11 += edge; a22 += edge;
            float2 d = edge * (base - flow_at(BaseFlow,up)); b1 -= d.x; b2 -= d.y;
        }
    }
    A11[p] = a11; A12[p] = a12; A22[p] = a22; B1[p] = b1; B2[p] = b2;
    Weights[p] = wcur;
}
