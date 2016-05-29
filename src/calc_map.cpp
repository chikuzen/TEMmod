/*
  calc_map.cpp

  This file is part of TEMmod

  Copyright (C) 2013 Oka Motofumi

  Authors: Oka Motofumi (chikuzen.mo at gmail dot com)

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
*/


#include <string.h>
#include <math.h>
#include "temmod.h"
#include "simd.h"

static inline void line_copy(uint8_t* dstp, const uint8_t* srcp, int width)
{
    memcpy(dstp, srcp, width);
    dstp[-1] = dstp[0];
    dstp[-2] = dstp[0];
    dstp[width] = dstp[width - 1];
    dstp[width + 1] = dstp[width - 1];
}


/*
 https://en.wikipedia.org/wiki/Alpha_max_plus_beta_min_algorithm
 sqrt(I*I + Q*Q) -> max(abs(I), abs(Q)) * 15/16 + min(abs(I), abs(Q)) * 15/32
 largest error: 6.25%  mean error: 1.88% ...... not too bad.
*/

static const __declspec(align(16)) int16_t fifteens[8] = {15, 15, 15, 15,
                                                          15, 15, 15, 15};

static void __stdcall
calc_map_3(const uint8_t* srcp, uint8_t* dstp, uint8_t* buff, int src_pitch,
           int dst_pitch, int buff_pitch, int width, int height,
           int threshold, float sc)
{
    uint8_t* p0 = buff + 16;
    uint8_t* p1 = p0 + buff_pitch;
    uint8_t* p2 = p1 + buff_pitch;
    uint8_t* orig = p0;
    uint8_t* end = p2;

    line_copy(p0, srcp, width);
    line_copy(p1, srcp, width);
    srcp += src_pitch;

    __m128i xscth;
    if (threshold == 0) {
        int16_t scale = sc > 0 ? (int16_t)(sc * (1 << 8) + 0.5) : 1 << 8;
        xscth = _mm_set1_epi16(scale);
    } else {
        xscth = _mm_set1_epi16(static_cast<int16_t>(threshold));
    }

    const __m128i zero = _mm_setzero_si128();
    const __m128i ab = _mm_set1_epi16(15);

    for (int y = 0; y < height; y++) {
        line_copy(p2, srcp, width);
        for (int x = 0; x < width; x += 16) {
            __m128i xmm0 = absdiff_u8(loadu(p1 + x - 1), loadu(p1 + x + 1));
            __m128i xmm1 = absdiff_u8(load(p0 + x), load(p2 + x));

            __m128i xmax = max_u8(xmm0, xmm1);
            __m128i xmin = min_u8(xmm0, xmm1);

            xmm0 = rshift_i16(mullo_i16(ab, unpacklo_i8(xmax, zero)), 4);
            xmm1 = rshift_i16(mullo_i16(ab, unpackhi_i8(xmax, zero)), 4);
            __m128i t0 = rshift_i16(mullo_i16(ab, unpacklo_i8(xmin, zero)), 5);
            __m128i t1 = rshift_i16(mullo_i16(ab, unpackhi_i8(xmin, zero)), 5);
            
            xmm0 = _mm_adds_epu16(xmm0, t0);
            xmm1 = _mm_adds_epu16(xmm1, t1);

            if (threshold != 0) {
                xmm0 = cmpgt_i16(xmm0, xscth);
                xmm1 = cmpgt_i16(xmm1, xscth);
                store(dstp + x, packs_i16(xmm0, xmm1));
                continue;
            }
            t0 = madd_i16(xscth, unpacklo_i16(xmm0, zero));
            t1 = madd_i16(xscth, unpackhi_i16(xmm0, zero));
            xmm0 = packus_i32(rshift_i32(t0, 8), rshift_i32(t1, 8));
            t0 = madd_i16(xscth, unpacklo_i16(xmm1, zero));
            t1 = madd_i16(xscth, unpackhi_i16(xmm1, zero));
            xmm1 = packus_i32(rshift_i32(t0, 8), rshift_i32(t1, 8));
            store(dstp + x, packus_i16(xmm0, xmm1));
        }
        p0 = p1;
        p1 = p2;
        p2 = p2 == end ? orig : p2 + buff_pitch;
        dstp += dst_pitch;
        srcp += src_pitch * (y < height - 2 ? 1 : 0);
    }
}


/*
    sqrt((Ix*Ix+Iy*Iy)*0.0001)*(255.0/158.1)*3
  = sqrt(Ix*Ix + Iy*Iy) * (255.0/158.1*0.01*3)
  = (max(abs(Ix),abs(Iy))*15/16 + min(abs(Ix),abs(Iy))*15/32)*(255.0/158.1*0.01*3*(1<<16))>>16
*/

static const __declspec(align(16)) int16_t ar_32767[] = {
    32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767
};
// 12:74 -> (int)(12.0/3):(int)(74.0/3+0.5)
static const __declspec(align(16)) int16_t ar_mulx[][8] = {
    {  4,   4,   4,   4,   4,   4,   4,   4},
    {-25, -25, -25, -25, -25, -25, -25, -25},
    { 25,  25,  25,  25,  25,  25,  25,  25},
    { -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4}
};
static const __declspec(align(16)) int16_t ar_muly[][8] = {
    { -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4},
    { 25,  25,  25,  25,  25,  25,  25,  25},
    {-25, -25, -25, -25, -25, -25, -25, -25},
    {  4,   4,   4,   4,   4,   4,   4,   4}
};
#undef SCALE

static void __stdcall
calc_map_4(const uint8_t* srcp, uint8_t* dstp, uint8_t* buff, int src_pitch,
           int dst_pitch, int buff_pitch, int width, int height,
           int threshold, float sc)
{
    uint8_t* p0 = buff + 16;
    uint8_t* p1 = p0 + buff_pitch;
    uint8_t* p2 = p1 + buff_pitch;
    uint8_t* p3 = p2 + buff_pitch;
    uint8_t* p4 = p3 + buff_pitch;
    uint8_t* orig = p0;
    uint8_t* end = p4;

    line_copy(p0, srcp, width);
    line_copy(p1, srcp, width);
    line_copy(p2, srcp, width);
    srcp += src_pitch;
    line_copy(p3, srcp, width);
    srcp += src_pitch;

    int16_t scale = sc > 0 ? (int16_t)(sc * 0.01 * 3 * (1 << 16) + 0.5) :
                             (int16_t)(255.0 / 158.1 * 0.01 * 3 * (1 << 16) + 0.5);
    __declspec(align(16)) int16_t ar_thresh[8];
    __declspec(align(16)) int16_t ar_scale[8];
    for (int i = 0; i < 8; i++) {
        ar_thresh[i] = threshold - 32767;
        ar_scale[i] = scale;
    }

    for (int y = 0; y < height; y++) {
        line_copy(p4, srcp, width);
        for (int x = 0; x < width; x += 16) {

            int posh[] = {- 2, - 1, 1, 2};
            uint8_t* posv[] = {p0, p1, p3, p4};
            __m128i zero = _mm_setzero_si128();
            __m128i sumx[2] = {zero, zero};
            __m128i sumy[2] = {zero, zero};

            for (int i = 0; i < 4; i++) {
                __m128i xmm0, xmm1, xmul;
                xmul = _mm_load_si128((__m128i*)ar_mulx[i]);
                xmm0 = _mm_loadu_si128((__m128i*)(p2 + x + posh[i]));
                xmm1 = _mm_unpackhi_epi8(xmm0, zero);
                xmm0 = _mm_unpacklo_epi8(xmm0, zero);
                sumx[0] = _mm_add_epi16(sumx[0], _mm_mullo_epi16(xmm0, xmul));
                sumx[1] = _mm_add_epi16(sumx[1], _mm_mullo_epi16(xmm1, xmul));

                xmul = _mm_load_si128((__m128i*)ar_muly[i]);
                xmm0 = _mm_load_si128((__m128i*)(posv[i] + x));
                xmm1 = _mm_unpackhi_epi8(xmm0, zero);
                xmm0 = _mm_unpacklo_epi8(xmm0, zero);
                sumy[0] = _mm_add_epi16(sumy[0], _mm_mullo_epi16(xmm0, xmul));
                sumy[1] = _mm_add_epi16(sumy[1], _mm_mullo_epi16(xmm1, xmul));
            }

            __m128i ab = _mm_load_si128((__m128i*)fifteens);
            for (int i = 0; i < 2; i++) {
                __m128i max, min, mull, mulh;
                sumx[i] = abs_i16(sumx[i]);
                sumy[i] = abs_i16(sumy[i]);
                max = _mm_max_epi16(sumx[i], sumy[i]);
                min = _mm_min_epi16(sumx[i], sumy[i]);

                mull = _mm_srli_epi32(_mm_madd_epi16(ab, _mm_unpacklo_epi16(max, zero)), 4);
                mulh = _mm_srli_epi32(_mm_madd_epi16(ab, _mm_unpackhi_epi16(max, zero)), 4);
                max = packus_i32(mull, mulh);

                mull = _mm_srli_epi32(_mm_madd_epi16(ab, _mm_unpacklo_epi16(min, zero)), 5);
                mulh = _mm_srli_epi32(_mm_madd_epi16(ab, _mm_unpackhi_epi16(min, zero)), 5);
                min = packus_i32(mull, mulh);

                sumx[i] = _mm_adds_epu16(max, min);
            }

            if (threshold > 0) {
                __m128i xthr = _mm_load_si128((__m128i*)ar_thresh);
                __m128i xsub = _mm_load_si128((__m128i*)ar_32767);
                sumx[0] = _mm_cmpgt_epi16(_mm_sub_epi16(sumx[0], xsub), xthr);
                sumx[1] = _mm_cmpgt_epi16(_mm_sub_epi16(sumx[1], xsub), xthr);
                _mm_store_si128((__m128i*)(dstp + x), _mm_packs_epi16(sumx[0], sumx[1]));
                continue;
            }

            __m128i scale = _mm_load_si128((__m128i*)ar_scale);
            for (int i = 0; i < 2; i++) {
                __m128i mull = _mm_madd_epi16(scale, _mm_unpacklo_epi16(sumx[i], zero));
                __m128i mulh = _mm_madd_epi16(scale, _mm_unpackhi_epi16(sumx[i], zero));
                sumx[i] = packus_i32(_mm_srli_epi32(mull, 16), _mm_srli_epi32(mulh, 16));
            }
            _mm_store_si128((__m128i*)(dstp + x), _mm_packus_epi16(sumx[0], sumx[1]));
        }
        srcp += src_pitch * (y < height - 3);
        dstp += dst_pitch;
        p0 = p1;
        p1 = p2;
        p2 = p3;
        p3 = p4;
        p4 = (p4 == end) ? orig : p4 + buff_pitch;
    }
}

/* generic Sobel edge detection */
static void __stdcall
calc_map_5(const uint8_t* srcp, uint8_t* dstp, uint8_t* buff, int src_pitch,
           int dst_pitch, int buff_pitch, int width, int height,
           int threshold, float sc)
{
    uint8_t* p0 = buff + 16;
    uint8_t* p1 = p0 + buff_pitch;
    uint8_t* p2 = p1 + buff_pitch;
    uint8_t* orig = p0;
    uint8_t* end = p2;

    line_copy(p0, srcp, width);
    line_copy(p1, srcp, width);
    srcp += src_pitch;

    int16_t scale = (int16_t)((sc > 0 ? sc : 0.25) * (1 << 8) + 0.5);

    __declspec(align(16)) int16_t ar_thresh[8];
    __declspec(align(16)) int16_t ar_scale[8];
    for (int i = 0; i < 8; i++) {
        ar_thresh[i] = (int16_t)threshold;
        ar_scale[i] = scale;
    }

    for (int y = 0; y < height; y++) {
        line_copy(p2, srcp, width);
        uint8_t *array[][6] = {
            /*  -1      -1       -2       1       1       2   */
            { p0 - 1, p2 - 1, p1 - 1, p0 + 1, p2 + 1, p1 + 1 },
            { p0 - 1, p0 + 1, p0    , p2 - 1, p2 + 1, p2     }
        };

        for (int x = 0; x < width; x += 16) {
            __m128i zero = _mm_setzero_si128();
            __m128i sumlo[2], sumhi[2];

            sumlo[1] = _mm_loadu_si128((__m128i *)(p0 + x - 1));
            sumlo[0] = _mm_unpacklo_epi8(sumlo[1], zero);
            sumhi[0] = _mm_unpackhi_epi8(sumlo[1], zero);
            sumlo[1] = sumlo[0];
            sumhi[1] = sumhi[0];

            for (int i = 0; i < 2; i++) {
                __m128i xmm0, xmm1, all1, one;

                xmm0 = _mm_loadu_si128((__m128i *)(array[i][1] + x));
                xmm1 = _mm_unpackhi_epi8(xmm0, zero);
                xmm0 = _mm_unpacklo_epi8(xmm0, zero);
                sumlo[i] = _mm_add_epi16(sumlo[i], xmm0);
                sumhi[i] = _mm_add_epi16(sumhi[i], xmm1);

                xmm0 = _mm_loadu_si128((__m128i *)(array[i][2] + x));
                xmm1 = _mm_slli_epi16(_mm_unpackhi_epi8(xmm0, zero), 1);
                xmm0 = _mm_slli_epi16(_mm_unpacklo_epi8(xmm0, zero), 1);
                sumlo[i] = _mm_add_epi16(sumlo[i], xmm0);
                sumhi[i] = _mm_add_epi16(sumhi[i], xmm1);

                // -x - y - 2z = (x + y + 2z) * -1
                all1 = _mm_cmpeq_epi32(xmm0, xmm0);
                one = _mm_srli_epi16(xmm0, 15);
                sumlo[i] = _mm_add_epi16(one, _mm_xor_si128(sumlo[i], all1));
                sumhi[i] = _mm_add_epi16(one, _mm_xor_si128(sumhi[i], all1));

                xmm0 = _mm_loadu_si128((__m128i *)(array[i][3] + x));
                xmm1 = _mm_unpackhi_epi8(xmm0, zero);
                xmm0 = _mm_unpacklo_epi8(xmm0, zero);
                sumlo[i] = _mm_add_epi16(sumlo[i], xmm0);
                sumhi[i] = _mm_add_epi16(sumhi[i], xmm1);

                xmm0 = _mm_loadu_si128((__m128i *)(array[i][4] + x));
                xmm1 = _mm_unpackhi_epi8(xmm0, zero);
                xmm0 = _mm_unpacklo_epi8(xmm0, zero);
                sumlo[i] = _mm_add_epi16(sumlo[i], xmm0);
                sumhi[i] = _mm_add_epi16(sumhi[i], xmm1);

                xmm0 = _mm_loadu_si128((__m128i *)(array[i][5] + x));
                xmm1 = _mm_slli_epi16(_mm_unpackhi_epi8(xmm0, zero), 1);
                xmm0 = _mm_slli_epi16(_mm_unpacklo_epi8(xmm0, zero), 1);
                sumlo[i] = _mm_add_epi16(sumlo[i], xmm0);
                sumhi[i] = _mm_add_epi16(sumhi[i], xmm1);

                xmm0 = _mm_add_epi16(one, _mm_xor_si128(sumlo[i], all1));
                sumlo[i] = _mm_or_si128(_mm_max_epi16(sumlo[i], zero),
                                        _mm_max_epi16(xmm0, zero));

                xmm0 = _mm_add_epi16(one, _mm_xor_si128(sumhi[i], all1));
                sumhi[i] = _mm_or_si128(_mm_max_epi16(sumhi[i], zero),
                                        _mm_max_epi16(xmm0, zero));
            }

            __m128i xmul = _mm_load_si128((__m128i *)fifteens);

            __m128i maxlh = _mm_max_epi16(sumlo[0], sumlo[1]);
            __m128i minlh = _mm_min_epi16(sumlo[0], sumlo[1]);
            maxlh = _mm_srli_epi16(_mm_mullo_epi16(maxlh, xmul), 4);
            minlh = _mm_srli_epi16(_mm_mullo_epi16(minlh, xmul), 5);
            __m128i outlo = _mm_add_epi16(maxlh, minlh);

            maxlh = _mm_max_epi16(sumhi[0], sumhi[1]);
            minlh = _mm_min_epi16(sumhi[0], sumhi[1]);
            maxlh = _mm_srli_epi16(_mm_mullo_epi16(maxlh, xmul), 4);
            minlh = _mm_srli_epi16(_mm_mullo_epi16(minlh, xmul), 5);
            __m128i outhi = _mm_add_epi16(maxlh, minlh);

            if (threshold == 0) {
                xmul = _mm_load_si128((__m128i*)ar_scale);
                __m128i t0 = _mm_madd_epi16(xmul, _mm_unpacklo_epi16(outlo, zero));
                __m128i t1 = _mm_madd_epi16(xmul, _mm_unpackhi_epi16(outlo, zero));
                outlo = packus_i32(_mm_srli_epi32(t0, 8), _mm_srli_epi32(t1, 8));
                t0 = _mm_madd_epi16(xmul, _mm_unpacklo_epi16(outhi, zero));
                t1 = _mm_madd_epi16(xmul, _mm_unpackhi_epi16(outhi, zero));
                outhi = packus_i32(_mm_srli_epi32(t0, 8), _mm_srli_epi32(t1, 8));
                outlo = _mm_packus_epi16(outlo, outhi);
                _mm_store_si128((__m128i*)(dstp + x), outlo);
                continue;
            }

            __m128i xthr = _mm_load_si128((__m128i*)ar_thresh);
            outlo = _mm_cmpgt_epi16(outlo, xthr);
            outhi = _mm_cmpgt_epi16(outhi, xthr);
            _mm_store_si128((__m128i*)(dstp + x), _mm_packs_epi16(outlo, outhi));
        }
        p0 = p1;
        p1 = p2;
        p2 = p2 == end ? orig : p2 + buff_pitch;
        dstp += dst_pitch;
        srcp += src_pitch * (y < height - 2 ? 1 : 0);
    }
}


static void __stdcall
calc_map_1(const uint8_t* srcp, uint8_t* dstp, uint8_t* buff, int src_pitch,
           int dst_pitch, int buff_pitch, int width, int height,
           int threshold, float sc)
{
    const uint8_t *p0 = srcp;
    const uint8_t *p1 = p0 + src_pitch;
    const uint8_t *p2 = p1 + src_pitch;
    const float scale = sc > 0 ? sc : (float)(255.0 / 127.5);

    memset(dstp, 0, dst_pitch);
    dstp += dst_pitch;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int ix = p1[x + 1] - p1[x - 1];
            int iy = p0[x] - p2[x];
            int temp = (int)(sqrt((ix * ix + iy * iy) * 0.25) * scale + 0.5); 
            if (temp > 255) temp = 255;
            dstp[x] = temp;
        }
        p0 += src_pitch;
        p1 += src_pitch;
        p2 += src_pitch;
        dstp += dst_pitch;
    }
    memset(dstp, 0, dst_pitch);
}


static void __stdcall
calc_map_1t(const uint8_t* srcp, uint8_t* dstp, uint8_t* buff, int src_pitch,
            int dst_pitch, int buff_pitch, int width, int height,
            int threshold, float scale)
{
    const uint8_t *p0 = srcp;
    const uint8_t *p1 = p0 + src_pitch;
    const uint8_t *p2 = p1 + src_pitch;

    memset(dstp, 0, dst_pitch);
    dstp += dst_pitch;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int ix = p1[x + 1] - p1[x - 1];
            int iy = p0[x] - p2[x];
            dstp[x] =  (ix * ix + iy * iy > threshold) * 255;
        }
        p0 += src_pitch;
        p1 += src_pitch;
        p2 += src_pitch;
        dstp += dst_pitch;
    }
    memset(dstp, 0, dst_pitch);
}


static void __stdcall
calc_map_2(const uint8_t* srcp, uint8_t* dstp, uint8_t* buff, int src_pitch,
            int dst_pitch, int buff_pitch, int width, int height,
            int threshold, float sc)
{
    const uint8_t *p0 = srcp;
    const uint8_t *p1 = p0 + src_pitch;
    const uint8_t *p2 = p1 + src_pitch;
    const uint8_t *p3 = p2 + src_pitch;
    const uint8_t *p4 = p3 + src_pitch;
    const float scale = sc > 0 ? sc : (float)(255.0 / 158.1);

    memset(dstp, 0, dst_pitch * 2);
    dstp += 2 * dst_pitch;
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            int ix = 12 * (p2[x - 2] - p2[x + 2]) + 74 * (p2[x + 1] - p2[x - 1]);
            int iy = 12 * (p4[x] - p0[x]) + 74 * (p1[x] - p3[x]);
            int temp = (int)(sqrt((ix * ix + iy * iy) * 0.0001f) * scale + 0.5f);
            dstp[x] =  temp > 255 ? 255 : temp;
        }
        p0 += src_pitch;
        p1 += src_pitch;
        p2 += src_pitch;
        p3 += src_pitch;
        p4 += src_pitch;
        dstp += dst_pitch;
    }
    memset(dstp, 0, dst_pitch * 2);
}


static void __stdcall
calc_map_2t(const uint8_t* srcp, uint8_t* dstp, uint8_t* buff, int src_pitch,
            int dst_pitch, int buff_pitch, int width, int height,
            int threshold, float scale)
{
    const uint8_t *p0 = srcp;
    const uint8_t *p1 = p0 + src_pitch;
    const uint8_t *p2 = p1 + src_pitch;
    const uint8_t *p3 = p2 + src_pitch;
    const uint8_t *p4 = p3 + src_pitch;

    memset(dstp, 0, dst_pitch * 2);
    dstp += 2 * dst_pitch;
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            int ix = 12 * (p2[x - 2] - p2[x + 2]) + 74 * (p2[x + 1] - p2[x - 1]);
            int iy = 12 * (p4[x] - p0[x]) + 74 * (p1[x] - p3[x]);
            dstp[x] =  (ix * ix + iy * iy > threshold) * 255;
        }
        p0 += src_pitch;
        p1 += src_pitch;
        p2 += src_pitch;
        p3 += src_pitch;
        p4 += src_pitch;
        dstp += dst_pitch;
    }
    memset(dstp, 0, dst_pitch * 2);
}

const calc_map_func calc_maps[] = {
    calc_map_1, calc_map_1t,
    calc_map_2, calc_map_2t,
    calc_map_3,
    calc_map_4,
    calc_map_5
};
