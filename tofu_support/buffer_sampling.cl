
/* OpenCL linear sampling: pixel centres are at integer + 0.5.
   NONE is undefined outside an image in OpenCL. We explicitly use zero
   outside the measured detector, matching the reconstruction convention.
   CLAMP uses a zero border; CLAMP_TO_EDGE repeats the edge pixel. */
inline int buffer_address(int i, int n, int mode) {
    if (mode == 0x1131) return clamp(i, 0, n - 1);
    if (mode == 0x1133) return ((i % n) + n) % n;
    if (mode == 0x1134) {
        int q = ((i % (2 * n)) + 2 * n) % (2 * n);
        return q < n ? q : 2 * n - 1 - q;
    }
    return i;
}
inline float buffer_texel(global const float *p, int x, int y, int4 info) {
    x = buffer_address(x, info.x, info.z);
    y = buffer_address(y, info.y, info.z);
    if (x < 0 || y < 0 || x >= info.x || y >= info.y) return 0.0f;
    return p[y * info.x + x];
}
inline float buffer_read_linear(global const float *p, int4 info, float2 coord) {
    float2 q = coord - (float2)(0.5f, 0.5f);
    int x = (int)floor(q.x), y = (int)floor(q.y);
    float a = q.x - floor(q.x), b = q.y - floor(q.y);
    float v00 = buffer_texel(p, x, y, info);
    float v10 = buffer_texel(p, x + 1, y, info);
    float v01 = buffer_texel(p, x, y + 1, info);
    float v11 = buffer_texel(p, x + 1, y + 1, info);
    return (1.0f-b) * ((1.0f-a)*v00 + a*v10) + b * ((1.0f-a)*v01 + a*v11);
}
