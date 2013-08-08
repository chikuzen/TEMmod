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
#include <emmintrin.h>
#include "temmod.h"

static inline void line_copy(uint8_t* dstp, const uint8_t* srcp, int width)
{
    memcpy(dstp, srcp, width);
    dstp[-1] = dstp[0];
    dstp[-2] = dstp[0];
    dstp[width] = dstp[width - 1];
    dstp[width + 1] = dstp[width - 1];
}


static inline __m128i mm_abs_epi32(__m128i in)
{
    __m128i all1 = _mm_cmpeq_epi32(in, in);
    __m128i mask = _mm_cmpgt_epi32(in, _mm_setzero_si128());
    __m128i temp = _mm_add_epi32(_mm_xor_si128(in, all1),
                                 _mm_srli_epi32(all1, 31));
    return _mm_or_si128(_mm_and_si128(mask, in),
                        _mm_andnot_si128(mask, temp));
}


static inline __m128i mm_max_epi32(__m128i xmm0, __m128i xmm1)
{
    __m128i mask = _mm_cmpgt_epi32(xmm0, xmm1);
    return _mm_or_si128(_mm_and_si128(mask, xmm0),
                        _mm_andnot_si128(mask, xmm1));
}


static inline __m128i mm_min_epi32(__m128i xmm0, __m128i xmm1)
{
    __m128i mask = _mm_cmplt_epi32(xmm0, xmm1);
    return _mm_or_si128(_mm_and_si128(mask, xmm0),
                        _mm_andnot_si128(mask, xmm1));
}


/*
 https://en.wikipedia.org/wiki/Alpha_max_plus_beta_min_algorithm
 sqrt(I*I + Q*Q) ? max(abs(I), abs(Q)) * 0.96043387 + min(abs(I), abs(Q)) * 0.37982473
*/

#define ALPHA ((int32_t)(0.96043387 * (1 << 15) + 0.5))
#define BETA ((int32_t)(0.37982473 * (1 << 15) + 0.5))
static const __declspec(align(16)) int32_t ar_alpha1[] = {ALPHA, ALPHA, ALPHA, ALPHA};
static const __declspec(align(16)) int32_t ar_beta1[] = {BETA, BETA, BETA, BETA};
#undef ALPHA
#undef BETA

static void __stdcall
calc_map_1(const uint8_t* srcp, uint8_t* dstp, uint8_t* buff, int src_pitch,
           int dst_pitch, int buff_pitch, int width, int height, int16_t threshold)
{
    uint8_t* p0 = buff + 16;
    uint8_t* p1 = p0 + buff_pitch;
    uint8_t* p2 = p1 + buff_pitch;
    uint8_t* orig = p0;
    uint8_t* end = p2;

    line_copy(p0, srcp, width);
    line_copy(p1, srcp, width);
    srcp += src_pitch;

    __m128i xthr = _mm_set1_epi16(threshold);

    for (int y = 0; y < height; y++) {
        line_copy(p2, srcp, width);
        for (int x = 0; x < width; x += 16) {
            __m128i xmm0, xmm1, xmm2, xmm3, xmax, xmin, temp, ab, zero;

            xmm0 = _mm_loadu_si128((__m128i*)(p1 + x - 1));
            xmm1 = _mm_loadu_si128((__m128i*)(p1 + x + 1));
            xmm0 = _mm_subs_epu8(_mm_max_epu8(xmm0, xmm1),
                                 _mm_min_epu8(xmm0, xmm1));
            
            xmm1 = _mm_load_si128((__m128i*)(p0 + x));
            xmm2 = _mm_load_si128((__m128i*)(p2 + x));
            xmm1 = _mm_subs_epu8(_mm_max_epu8(xmm1, xmm2),
                                 _mm_min_epu8(xmm1, xmm2));

            xmax = _mm_max_epu8(xmm0, xmm1);
            xmin = _mm_min_epu8(xmm0, xmm1);

            zero = _mm_setzero_si128();
            ab = _mm_load_si128((__m128i*)ar_alpha1);

            temp = _mm_unpacklo_epi8(xmax, zero);
            xmm0 = _mm_madd_epi16(ab, _mm_unpacklo_epi16(temp, zero));
            xmm1 = _mm_madd_epi16(ab, _mm_unpackhi_epi16(temp, zero));
            temp = _mm_unpackhi_epi8(xmax, zero);
            xmm2 = _mm_madd_epi16(ab, _mm_unpacklo_epi16(temp, zero));
            xmm3 = _mm_madd_epi16(ab, _mm_unpackhi_epi16(temp, zero));

            ab = _mm_load_si128((__m128i*)ar_beta1);

            temp = _mm_unpacklo_epi8(xmin, zero);
            xmm0 = _mm_add_epi32(xmm0, _mm_madd_epi16(ab, _mm_unpacklo_epi16(temp, zero)));
            xmm1 = _mm_add_epi32(xmm1, _mm_madd_epi16(ab, _mm_unpackhi_epi16(temp, zero)));
            temp = _mm_unpackhi_epi8(xmin, zero);
            xmm2 = _mm_add_epi32(xmm2, _mm_madd_epi16(ab, _mm_unpacklo_epi16(temp, zero)));
            xmm3 = _mm_add_epi32(xmm3, _mm_madd_epi16(ab, _mm_unpackhi_epi16(temp, zero)));

            xmm0 = _mm_packs_epi32(_mm_srli_epi32(xmm0, 15), _mm_srli_epi32(xmm1, 15));
            xmm1 = _mm_packs_epi32(_mm_srli_epi32(xmm2, 15), _mm_srli_epi32(xmm3, 15));

            if (threshold == 0) {
                _mm_store_si128((__m128i*)(dstp + x), _mm_packus_epi16(xmm0, xmm1));
                continue;
            }

            xmm0 = _mm_cmpgt_epi16(xmm0, xthr);
            xmm1 = _mm_cmpgt_epi16(xmm1, xthr);
            _mm_store_si128((__m128i*)(dstp + x), _mm_packs_epi16(xmm0, xmm1));
        }
        p0 = p1;
        p1 = p2;
        p2 = p2 == end ? orig : p2 + buff_pitch;
        dstp += dst_pitch;
        srcp += src_pitch * (y < height - 2 ? 1 : 0);
    }
}

/*
 (sqrt((Ix*Ix+Iy*Iy)*0.0001)*1.612903)
    = (sqrt(Ix*Ix + Iy*Iy) * 0.01612903)
    = (max(abs(Ix), abs(Iy)) * 0.96043387 + min(abs(Ix), abs(Iy)) * 0.37982473)*0.01612903)
*/

#define ALPHA ((float)0.96043387)
#define BETA ((float)0.37982473)
#define SCALE ((float)(255.0 / 158.1 * 0.01))
static const __declspec(align(16)) float ar_alpha2[] = {ALPHA, ALPHA,ALPHA,ALPHA};
static const __declspec(align(16)) float ar_beta2[] = {BETA, BETA, BETA, BETA};
static const __declspec(align(16)) float ar_scale[] = {SCALE, SCALE, SCALE, SCALE};
static const __declspec(align(16)) int16_t ar_mulx[][8] = {
    { 12,  12,  12 , 12,  12,  12,  12,  12},
    {-74, -74, -74, -74, -74, -74, -74, -74},
    { 74,  74,  74,  74,  74,  74,  74,  74},
    {-12, -12, -12, -12, -12, -12, -12, -12}
};
static const __declspec(align(16)) int16_t ar_muly[][8] = {
    {-12, -12, -12, -12, -12, -12, -12, -12},
    { 74,  74,  74,  74,  74,  74,  74,  74},
    {-74, -74, -74, -74, -74, -74, -74, -74},
    { 12,  12,  12 , 12,  12,  12,  12,  12}
};
#undef ALPHA
#undef BETA
#undef SCALE

static void __stdcall
calc_map_2(const uint8_t* srcp, uint8_t* dstp, uint8_t* buff, int src_pitch,
           int dst_pitch, int buff_pitch, int width, int height, int16_t threshold)
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

    for (int y = 0; y < height; y++) {
        line_copy(p4, srcp, width);
        for (int x = 0; x < width; x += 16) {

            int posh[] = {- 2, - 1, 1, 2};
            uint8_t* posv[] = {p0, p1, p3, p4};
            __m128i zero = _mm_setzero_si128();
            __m128i sumx[4] = {zero, zero, zero, zero};
            __m128i sumy[4] = {zero, zero, zero, zero};

            for (int i = 0; i < 4; i++) {
                __m128i xmm0, xmm1, xmm2, xmul;
                xmul = _mm_load_si128((__m128i*)ar_mulx[i]);
                xmm0 = _mm_loadu_si128((__m128i*)(p2 + x + posh[i]));
                xmm1 = _mm_unpackhi_epi8(xmm0, zero);
                xmm0 = _mm_unpacklo_epi8(xmm0, zero);
                xmm2 = _mm_unpackhi_epi16(xmm0, zero);
                xmm0 = _mm_unpacklo_epi16(xmm0, zero);
                sumx[0] = _mm_add_epi32(sumx[0], _mm_madd_epi16(xmm0, xmul));
                sumx[1] = _mm_add_epi32(sumx[1], _mm_madd_epi16(xmm2, xmul));
                xmm0 = _mm_unpacklo_epi16(xmm1, zero);
                xmm1 = _mm_unpackhi_epi16(xmm1, zero);
                sumx[2] = _mm_add_epi32(sumx[2], _mm_madd_epi16(xmm0, xmul));
                sumx[3] = _mm_add_epi32(sumx[3], _mm_madd_epi16(xmm1, xmul));

                xmul = _mm_load_si128((__m128i*)ar_muly[i]);
                xmm0 = _mm_load_si128((__m128i*)(posv[i] + x));
                xmm1 = _mm_unpackhi_epi8(xmm0, zero);
                xmm0 = _mm_unpacklo_epi8(xmm0, zero);
                xmm2 = _mm_unpackhi_epi16(xmm0, zero);
                xmm0 = _mm_unpacklo_epi16(xmm0, zero);
                sumy[0] = _mm_add_epi32(sumy[0], _mm_madd_epi16(xmm0, xmul));
                sumy[1] = _mm_add_epi32(sumy[1], _mm_madd_epi16(xmm2, xmul));
                xmm0 = _mm_unpacklo_epi16(xmm1, zero);
                xmm1 = _mm_unpackhi_epi16(xmm1, zero);
                sumy[2] = _mm_add_epi32(sumy[2], _mm_madd_epi16(xmm0, xmul));
                sumy[3] = _mm_add_epi32(sumy[3], _mm_madd_epi16(xmm1, xmul));
            }

            __m128 alpha = _mm_load_ps(ar_alpha2);
            __m128 beta = _mm_load_ps(ar_beta2);
            for (int i = 0; i < 4; i++) {
                __m128 max, min;
                sumx[i] = mm_abs_epi32(sumx[i]);
                sumy[i] = mm_abs_epi32(sumy[i]);
                max = _mm_cvtepi32_ps(mm_max_epi32(sumx[i], sumy[i]));
                min = _mm_cvtepi32_ps(mm_min_epi32(sumx[i], sumy[i]));
                sumx[i] = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(max, alpha),
                                                     _mm_mul_ps(min, beta)));
            }

            if (threshold > 0) {
                __m128i xthr = _mm_set1_epi16(threshold);
                sumx[0] = _mm_packs_epi32(sumx[0], sumx[1]);
                sumx[1] = _mm_packs_epi32(sumx[2], sumx[3]);
                sumx[0] = _mm_cmpgt_epi16(sumx[0], xthr);
                sumx[1] = _mm_cmpgt_epi16(sumx[1], xthr);
                _mm_store_si128((__m128i*)(dstp + x), _mm_packs_epi16(sumx[0], sumx[1]));
                continue;
            }

            __m128 scale = _mm_set1_ps((float)(255.0 / 158.1 * 0.01));
            for (int i = 0; i < 4; i++) {
                sumx[i] = _mm_cvtps_epi32(_mm_mul_ps(scale, _mm_cvtepi32_ps(sumx[i])));
            }
            sumx[0] = _mm_packs_epi32(sumx[0], sumx[1]);
            sumx[1] = _mm_packs_epi32(sumx[2], sumx[3]);
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


const calc_map_func calc_maps[] = {calc_map_1, calc_map_2};
