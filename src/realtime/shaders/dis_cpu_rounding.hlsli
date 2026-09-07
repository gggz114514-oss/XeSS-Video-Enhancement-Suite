// Diagnostic CPU round-to-nearest emulation. FP64 residual corrections preserve
// the CPU FP32 operator contract; they are not lower-precision approximations.
// Product performance and hardware support must be measured before promotion.
float cpu_divide(float a, float b)
{
    precise float estimate = a / b;
    precise double residual = (double)a - (double)estimate * (double)b;
    precise double corrected = (double)estimate + residual / (double)b;
    return (float)corrected;
}
float cpu_sqrt(float a)
{
    precise float estimate = sqrt(a);
    precise double residual = (double)a - (double)estimate * (double)estimate;
    precise double corrected = (double)estimate + residual / (2.0 * (double)estimate);
    return (float)corrected;
}
