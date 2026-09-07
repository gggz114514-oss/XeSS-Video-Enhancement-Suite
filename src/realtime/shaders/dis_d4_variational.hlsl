// Gate D4 diagnostic: explicit GPU variational-refinement stage.
// The constants are the OpenCV DIS Fast preset values: alpha=20, gamma=10,
// delta=5, epsilon=0.01. Five command-list ping-pong dispatches are issued by
// the probe. This is a correctness/semantic bridge; D6 owns final tuning.
Texture2D<float> Current : register(t0);
Texture2D<float> Previous : register(t1);
Texture2D<float2> FlowIn : register(t2);
RWTexture2D<float2> FlowOut : register(u0);
cbuffer Params : register(b0) { uint width; uint height; float alpha; float gamma; float delta; float epsilon; };

float sample_image(float x, float y)
{
    x = clamp(x, 0.0, (float)width - 1.001); y = clamp(y, 0.0, (float)height - 1.001);
    uint x0 = (uint)floor(x), y0 = (uint)floor(y), x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
    float fx = x - x0, fy = y - y0;
    float a = Previous.Load(int3(x0, y0, 0)), b = Previous.Load(int3(x1, y0, 0));
    float c = Previous.Load(int3(x0, y1, 0)), d = Previous.Load(int3(x1, y1, 0));
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
}
float average_at(int x, int y, float2 f)
{
    x = clamp(x, 0, (int)width - 1); y = clamp(y, 0, (int)height - 1);
    return 0.5 * (Current.Load(int3(x, y, 0)) + sample_image((float)x + f.x, (float)y + f.y));
}
float2 load_flow(int x, int y)
{
    x = clamp(x, 0, (int)width - 1); y = clamp(y, 0, (int)height - 1);
    return FlowIn.Load(int3(x, y, 0));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    int x = (int)id.x, y = (int)id.y; float2 f = load_flow(x, y);
    float center = average_at(x, y, f);
    float ix = 0.5 * (average_at(x + 1, y, f) - average_at(x - 1, y, f));
    float iy = 0.5 * (average_at(x, y + 1, f) - average_at(x, y - 1, f));
    float iz = sample_image((float)x + f.x, (float)y + f.y) - Current.Load(int3(x, y, 0));
    float zeta2 = 0.01;
    float deriv = ix * ix + iy * iy + zeta2;
    float data_weight = (delta * 0.5) / max(sqrt(iz * iz / deriv + epsilon * epsilon), 1e-6) / deriv;
    float a11 = data_weight * ix * ix + zeta2, a12 = data_weight * ix * iy;
    float a22 = data_weight * iy * iy + zeta2;
    float b1 = -data_weight * iz * ix, b2 = -data_weight * iz * iy;
    // Gradient-constancy contribution, using the same gamma/2 robust penalty
    // as OpenCV's data-term construction.
    float ixx = average_at(x + 1, y, f) - 2.0 * center + average_at(x - 1, y, f);
    float iyy = average_at(x, y + 1, f) - 2.0 * center + average_at(x, y - 1, f);
    float ixy = 0.25 * (average_at(x + 1, y + 1, f) - average_at(x + 1, y - 1, f) - average_at(x - 1, y + 1, f) + average_at(x - 1, y - 1, f));
    float ixz = 0.5 * ((sample_image((float)(x + 1) + f.x, (float)y + f.y) - Current.Load(int3(clamp(x + 1, 0, (int)width - 1), y, 0))) - (sample_image((float)(x - 1) + f.x, (float)y + f.y) - Current.Load(int3(clamp(x - 1, 0, (int)width - 1), y, 0))));
    float iyz = 0.5 * ((sample_image((float)x + f.x, (float)(y + 1) + f.y) - Current.Load(int3(x, clamp(y + 1, 0, (int)height - 1), 0))) - (sample_image((float)x + f.x, (float)(y - 1) + f.y) - Current.Load(int3(x, clamp(y - 1, 0, (int)height - 1), 0))));
    float n1 = ixx * ixx + ixy * ixy + zeta2, n2 = iyy * iyy + ixy * ixy + zeta2;
    float qx = ixz + ixx * f.x + ixy * f.y, qy = iyz + ixy * f.x + iyy * f.y;
    float grad_weight = (gamma * 0.5) / max(sqrt(qx * qx / n1 + qy * qy / n2 + epsilon * epsilon), 1e-6);
    a11 += grad_weight * (ixx * ixx / n1 + ixy * ixy / n2); a12 += grad_weight * (ixx * ixy / n1 + ixy * iyy / n2); a22 += grad_weight * (ixy * ixy / n1 + iyy * iyy / n2);
    b1 -= grad_weight * (ixx * ixz / n1 + ixy * iyz / n2); b2 -= grad_weight * (ixy * ixz / n1 + iyy * iyz / n2);
    float2 left = load_flow(x - 1, y), right = load_flow(x + 1, y), up = load_flow(x, y - 1), down = load_flow(x, y + 1);
    float smooth = alpha * 0.5 / max(sqrt(dot(right - left, right - left) * 0.25 + dot(down - up, down - up) * 0.25 + epsilon * epsilon), 1e-6);
    float2 neighbor = 0.25 * (left + right + up + down);
    float2 target;
    target.x = (b1 + smooth * neighbor.x - a12 * (neighbor.y - f.y)) / max(a11 + smooth, 1e-6);
    target.y = (b2 + smooth * neighbor.y - a12 * (neighbor.x - f.x)) / max(a22 + smooth, 1e-6);
    FlowOut[id.xy] = f + 0.4 * (target - f);
}
