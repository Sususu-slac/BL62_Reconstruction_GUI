
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

#define rotate()    pixel.x -= x_center.x; \
                    pixel.y -= y_center; \
                    pixel.x = pixel.x * cos_roll + pixel.y * sin_roll; \
                    pixel.y = -pixel.x * sin_roll + pixel.y * cos_roll; \
                    pixel.x += x_center.x; \
                    pixel.y += y_center;

/*
 * Copyright (C) 2015-2016 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

kernel void backproject_burst_1 (
global const float *projection_0,
                                    global float *volume,
                                    const int4 projection_info,
                                    const int3 real_size,
                                    const float2 x_center,
                                    const float y_center,
                                    const float2 x_region,
                                    const float2 y_region,
                                    const float2 z_region,
                                    const float2 lamino_region,
                                    const float2 roll_region,
                                    float sin_lamino,
                                    float cos_lamino,
                                    const float sines,
                                    const float cosines,
                                    const float norm_factor,
                                    float sin_roll,
                                    float cos_roll,
                                    const int cumulate)
{
    int idx = get_global_id (0);
    int idy = get_global_id (1);
    int idz = get_global_id (2);
    float result, tmp, tmp_x, tmp_y;
    float2 pixel;
    float3 voxel;

    if (idx < real_size.x && idy < real_size.y && idz < real_size.z) {
        voxel.x = mad((float) idx, x_region.y, x_region.x);
        voxel.y = mad((float) idy, y_region.y, y_region.x);
        sin_lamino = sincos (mad((float) idz, lamino_region.y, lamino_region.x), &cos_lamino);
        tmp = mad(z_region.x, sin_lamino, y_center);
        tmp_x = voxel.x * cos_lamino;
        tmp_y = -voxel.y * cos_lamino;

        pixel.x = mad(voxel.x, cosines, mad(voxel.y, sines, x_center.x));
        pixel.y = mad(tmp_x, sines, mad(tmp_y, cosines, tmp));
        rotate ();
        result = buffer_read_linear (projection_0, projection_info, pixel);


        if (cumulate) {
            volume[idz * real_size.x * real_size.y + idy * real_size.x + idx] += result * norm_factor;
        } else {
            volume[idz * real_size.x * real_size.y + idy * real_size.x + idx] = result * norm_factor;
        }
    }
}
/*
 * Copyright (C) 2015-2016 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

kernel void backproject_burst_2 (
global const float *projection_0,
global const float *projection_1,
                                    global float *volume,
                                    const int4 projection_info,
                                    const int3 real_size,
                                    const float2 x_center,
                                    const float y_center,
                                    const float2 x_region,
                                    const float2 y_region,
                                    const float2 z_region,
                                    const float2 lamino_region,
                                    const float2 roll_region,
                                    float sin_lamino,
                                    float cos_lamino,
                                    const float2 sines,
                                    const float2 cosines,
                                    const float norm_factor,
                                    float sin_roll,
                                    float cos_roll,
                                    const int cumulate)
{
    int idx = get_global_id (0);
    int idy = get_global_id (1);
    int idz = get_global_id (2);
    float result, tmp, tmp_x, tmp_y;
    float2 pixel;
    float3 voxel;

    if (idx < real_size.x && idy < real_size.y && idz < real_size.z) {
        voxel.x = mad((float) idx, x_region.y, x_region.x);
        voxel.y = mad((float) idy, y_region.y, y_region.x);
        sin_lamino = sincos (mad((float) idz, lamino_region.y, lamino_region.x), &cos_lamino);
        tmp = mad(z_region.x, sin_lamino, y_center);
        tmp_x = voxel.x * cos_lamino;
        tmp_y = -voxel.y * cos_lamino;

        pixel.x = mad(voxel.x, cosines.s0, mad(voxel.y, sines.s0, x_center.x));
        pixel.y = mad(tmp_x, sines.s0, mad(tmp_y, cosines.s0, tmp));
        rotate ();
        result = buffer_read_linear (projection_0, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s1, mad(voxel.y, sines.s1, x_center.x));
        pixel.y = mad(tmp_x, sines.s1, mad(tmp_y, cosines.s1, tmp));
        rotate ();
        result += buffer_read_linear (projection_1, projection_info, pixel);


        if (cumulate) {
            volume[idz * real_size.x * real_size.y + idy * real_size.x + idx] += result * norm_factor;
        } else {
            volume[idz * real_size.x * real_size.y + idy * real_size.x + idx] = result * norm_factor;
        }
    }
}
/*
 * Copyright (C) 2015-2016 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

kernel void backproject_burst_4 (
global const float *projection_0,
global const float *projection_1,
global const float *projection_2,
global const float *projection_3,
                                    global float *volume,
                                    const int4 projection_info,
                                    const int3 real_size,
                                    const float2 x_center,
                                    const float y_center,
                                    const float2 x_region,
                                    const float2 y_region,
                                    const float2 z_region,
                                    const float2 lamino_region,
                                    const float2 roll_region,
                                    float sin_lamino,
                                    float cos_lamino,
                                    const float4 sines,
                                    const float4 cosines,
                                    const float norm_factor,
                                    float sin_roll,
                                    float cos_roll,
                                    const int cumulate)
{
    int idx = get_global_id (0);
    int idy = get_global_id (1);
    int idz = get_global_id (2);
    float result, tmp, tmp_x, tmp_y;
    float2 pixel;
    float3 voxel;

    if (idx < real_size.x && idy < real_size.y && idz < real_size.z) {
        voxel.x = mad((float) idx, x_region.y, x_region.x);
        voxel.y = mad((float) idy, y_region.y, y_region.x);
        sin_lamino = sincos (mad((float) idz, lamino_region.y, lamino_region.x), &cos_lamino);
        tmp = mad(z_region.x, sin_lamino, y_center);
        tmp_x = voxel.x * cos_lamino;
        tmp_y = -voxel.y * cos_lamino;

        pixel.x = mad(voxel.x, cosines.s0, mad(voxel.y, sines.s0, x_center.x));
        pixel.y = mad(tmp_x, sines.s0, mad(tmp_y, cosines.s0, tmp));
        rotate ();
        result = buffer_read_linear (projection_0, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s1, mad(voxel.y, sines.s1, x_center.x));
        pixel.y = mad(tmp_x, sines.s1, mad(tmp_y, cosines.s1, tmp));
        rotate ();
        result += buffer_read_linear (projection_1, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s2, mad(voxel.y, sines.s2, x_center.x));
        pixel.y = mad(tmp_x, sines.s2, mad(tmp_y, cosines.s2, tmp));
        rotate ();
        result += buffer_read_linear (projection_2, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s3, mad(voxel.y, sines.s3, x_center.x));
        pixel.y = mad(tmp_x, sines.s3, mad(tmp_y, cosines.s3, tmp));
        rotate ();
        result += buffer_read_linear (projection_3, projection_info, pixel);


        if (cumulate) {
            volume[idz * real_size.x * real_size.y + idy * real_size.x + idx] += result * norm_factor;
        } else {
            volume[idz * real_size.x * real_size.y + idy * real_size.x + idx] = result * norm_factor;
        }
    }
}
/*
 * Copyright (C) 2015-2016 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

kernel void backproject_burst_8 (
global const float *projection_0,
global const float *projection_1,
global const float *projection_2,
global const float *projection_3,
global const float *projection_4,
global const float *projection_5,
global const float *projection_6,
global const float *projection_7,
                                    global float *volume,
                                    const int4 projection_info,
                                    const int3 real_size,
                                    const float2 x_center,
                                    const float y_center,
                                    const float2 x_region,
                                    const float2 y_region,
                                    const float2 z_region,
                                    const float2 lamino_region,
                                    const float2 roll_region,
                                    float sin_lamino,
                                    float cos_lamino,
                                    const float8 sines,
                                    const float8 cosines,
                                    const float norm_factor,
                                    float sin_roll,
                                    float cos_roll,
                                    const int cumulate)
{
    int idx = get_global_id (0);
    int idy = get_global_id (1);
    int idz = get_global_id (2);
    float result, tmp, tmp_x, tmp_y;
    float2 pixel;
    float3 voxel;

    if (idx < real_size.x && idy < real_size.y && idz < real_size.z) {
        voxel.x = mad((float) idx, x_region.y, x_region.x);
        voxel.y = mad((float) idy, y_region.y, y_region.x);
        sin_lamino = sincos (mad((float) idz, lamino_region.y, lamino_region.x), &cos_lamino);
        tmp = mad(z_region.x, sin_lamino, y_center);
        tmp_x = voxel.x * cos_lamino;
        tmp_y = -voxel.y * cos_lamino;

        pixel.x = mad(voxel.x, cosines.s0, mad(voxel.y, sines.s0, x_center.x));
        pixel.y = mad(tmp_x, sines.s0, mad(tmp_y, cosines.s0, tmp));
        rotate ();
        result = buffer_read_linear (projection_0, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s1, mad(voxel.y, sines.s1, x_center.x));
        pixel.y = mad(tmp_x, sines.s1, mad(tmp_y, cosines.s1, tmp));
        rotate ();
        result += buffer_read_linear (projection_1, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s2, mad(voxel.y, sines.s2, x_center.x));
        pixel.y = mad(tmp_x, sines.s2, mad(tmp_y, cosines.s2, tmp));
        rotate ();
        result += buffer_read_linear (projection_2, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s3, mad(voxel.y, sines.s3, x_center.x));
        pixel.y = mad(tmp_x, sines.s3, mad(tmp_y, cosines.s3, tmp));
        rotate ();
        result += buffer_read_linear (projection_3, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s4, mad(voxel.y, sines.s4, x_center.x));
        pixel.y = mad(tmp_x, sines.s4, mad(tmp_y, cosines.s4, tmp));
        rotate ();
        result += buffer_read_linear (projection_4, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s5, mad(voxel.y, sines.s5, x_center.x));
        pixel.y = mad(tmp_x, sines.s5, mad(tmp_y, cosines.s5, tmp));
        rotate ();
        result += buffer_read_linear (projection_5, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s6, mad(voxel.y, sines.s6, x_center.x));
        pixel.y = mad(tmp_x, sines.s6, mad(tmp_y, cosines.s6, tmp));
        rotate ();
        result += buffer_read_linear (projection_6, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s7, mad(voxel.y, sines.s7, x_center.x));
        pixel.y = mad(tmp_x, sines.s7, mad(tmp_y, cosines.s7, tmp));
        rotate ();
        result += buffer_read_linear (projection_7, projection_info, pixel);


        if (cumulate) {
            volume[idz * real_size.x * real_size.y + idy * real_size.x + idx] += result * norm_factor;
        } else {
            volume[idz * real_size.x * real_size.y + idy * real_size.x + idx] = result * norm_factor;
        }
    }
}
/*
 * Copyright (C) 2015-2016 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

kernel void backproject_burst_16 (
global const float *projection_0,
global const float *projection_1,
global const float *projection_2,
global const float *projection_3,
global const float *projection_4,
global const float *projection_5,
global const float *projection_6,
global const float *projection_7,
global const float *projection_8,
global const float *projection_9,
global const float *projection_10,
global const float *projection_11,
global const float *projection_12,
global const float *projection_13,
global const float *projection_14,
global const float *projection_15,
                                    global float *volume,
                                    const int4 projection_info,
                                    const int3 real_size,
                                    const float2 x_center,
                                    const float y_center,
                                    const float2 x_region,
                                    const float2 y_region,
                                    const float2 z_region,
                                    const float2 lamino_region,
                                    const float2 roll_region,
                                    float sin_lamino,
                                    float cos_lamino,
                                    const float16 sines,
                                    const float16 cosines,
                                    const float norm_factor,
                                    float sin_roll,
                                    float cos_roll,
                                    const int cumulate)
{
    int idx = get_global_id (0);
    int idy = get_global_id (1);
    int idz = get_global_id (2);
    float result, tmp, tmp_x, tmp_y;
    float2 pixel;
    float3 voxel;

    if (idx < real_size.x && idy < real_size.y && idz < real_size.z) {
        voxel.x = mad((float) idx, x_region.y, x_region.x);
        voxel.y = mad((float) idy, y_region.y, y_region.x);
        sin_lamino = sincos (mad((float) idz, lamino_region.y, lamino_region.x), &cos_lamino);
        tmp = mad(z_region.x, sin_lamino, y_center);
        tmp_x = voxel.x * cos_lamino;
        tmp_y = -voxel.y * cos_lamino;

        pixel.x = mad(voxel.x, cosines.s0, mad(voxel.y, sines.s0, x_center.x));
        pixel.y = mad(tmp_x, sines.s0, mad(tmp_y, cosines.s0, tmp));
        rotate ();
        result = buffer_read_linear (projection_0, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s1, mad(voxel.y, sines.s1, x_center.x));
        pixel.y = mad(tmp_x, sines.s1, mad(tmp_y, cosines.s1, tmp));
        rotate ();
        result += buffer_read_linear (projection_1, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s2, mad(voxel.y, sines.s2, x_center.x));
        pixel.y = mad(tmp_x, sines.s2, mad(tmp_y, cosines.s2, tmp));
        rotate ();
        result += buffer_read_linear (projection_2, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s3, mad(voxel.y, sines.s3, x_center.x));
        pixel.y = mad(tmp_x, sines.s3, mad(tmp_y, cosines.s3, tmp));
        rotate ();
        result += buffer_read_linear (projection_3, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s4, mad(voxel.y, sines.s4, x_center.x));
        pixel.y = mad(tmp_x, sines.s4, mad(tmp_y, cosines.s4, tmp));
        rotate ();
        result += buffer_read_linear (projection_4, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s5, mad(voxel.y, sines.s5, x_center.x));
        pixel.y = mad(tmp_x, sines.s5, mad(tmp_y, cosines.s5, tmp));
        rotate ();
        result += buffer_read_linear (projection_5, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s6, mad(voxel.y, sines.s6, x_center.x));
        pixel.y = mad(tmp_x, sines.s6, mad(tmp_y, cosines.s6, tmp));
        rotate ();
        result += buffer_read_linear (projection_6, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s7, mad(voxel.y, sines.s7, x_center.x));
        pixel.y = mad(tmp_x, sines.s7, mad(tmp_y, cosines.s7, tmp));
        rotate ();
        result += buffer_read_linear (projection_7, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s8, mad(voxel.y, sines.s8, x_center.x));
        pixel.y = mad(tmp_x, sines.s8, mad(tmp_y, cosines.s8, tmp));
        rotate ();
        result += buffer_read_linear (projection_8, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.s9, mad(voxel.y, sines.s9, x_center.x));
        pixel.y = mad(tmp_x, sines.s9, mad(tmp_y, cosines.s9, tmp));
        rotate ();
        result += buffer_read_linear (projection_9, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.sa, mad(voxel.y, sines.sa, x_center.x));
        pixel.y = mad(tmp_x, sines.sa, mad(tmp_y, cosines.sa, tmp));
        rotate ();
        result += buffer_read_linear (projection_10, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.sb, mad(voxel.y, sines.sb, x_center.x));
        pixel.y = mad(tmp_x, sines.sb, mad(tmp_y, cosines.sb, tmp));
        rotate ();
        result += buffer_read_linear (projection_11, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.sc, mad(voxel.y, sines.sc, x_center.x));
        pixel.y = mad(tmp_x, sines.sc, mad(tmp_y, cosines.sc, tmp));
        rotate ();
        result += buffer_read_linear (projection_12, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.sd, mad(voxel.y, sines.sd, x_center.x));
        pixel.y = mad(tmp_x, sines.sd, mad(tmp_y, cosines.sd, tmp));
        rotate ();
        result += buffer_read_linear (projection_13, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.se, mad(voxel.y, sines.se, x_center.x));
        pixel.y = mad(tmp_x, sines.se, mad(tmp_y, cosines.se, tmp));
        rotate ();
        result += buffer_read_linear (projection_14, projection_info, pixel);
        pixel.x = mad(voxel.x, cosines.sf, mad(voxel.y, sines.sf, x_center.x));
        pixel.y = mad(tmp_x, sines.sf, mad(tmp_y, cosines.sf, tmp));
        rotate ();
        result += buffer_read_linear (projection_15, projection_info, pixel);


        if (cumulate) {
            volume[idz * real_size.x * real_size.y + idy * real_size.x + idx] += result * norm_factor;
        } else {
            volume[idz * real_size.x * real_size.y + idy * real_size.x + idx] = result * norm_factor;
        }
    }
}

