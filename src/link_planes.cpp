/*
  link_planes.cpp

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


#include <emmintrin.h>
#include "temmod.h"


static void __stdcall link_y_to_uv_444(PVideoFrame& dst)
{
    const int sou = sizeof(unsigned);
    int pitch_y = dst->GetPitch(PLANAR_Y) / sou;
    int pitch_uv = dst->GetPitch(PLANAR_U) / sou;
    int width = ((dst->GetRowSize(PLANAR_U) + sou - 1) / sou) * sou;
    int height = dst->GetHeight(PLANAR_U);
    const unsigned* y0 = (unsigned*)dst->GetReadPtr(PLANAR_Y);
    unsigned* u0 = (unsigned*)dst->GetWritePtr(PLANAR_U);
    unsigned* v0 = (unsigned*)dst->GetWritePtr(PLANAR_V);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            u0[x] |= y0[x];
            v0[x] |= y0[x];
        }
        y0 += pitch_y;
        u0 += pitch_uv;
        v0 += pitch_uv;
    }
}


static void __stdcall link_all_444(PVideoFrame& dst)
{
    const int sou = sizeof(unsigned);
    int pitch_y = dst->GetPitch(PLANAR_Y) / sou;
    int pitch_uv = dst->GetPitch(PLANAR_U) / sou;
    int width = ((dst->GetRowSize(PLANAR_U) + sou - 1) / sou) * sou;
    int height = dst->GetHeight(PLANAR_U);
    unsigned* y0 = (unsigned*)dst->GetWritePtr(PLANAR_Y);
    unsigned* u0 = (unsigned*)dst->GetWritePtr(PLANAR_U);
    unsigned* v0 = (unsigned*)dst->GetWritePtr(PLANAR_V);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned samples = y0[x] | u0[x] | v0[x];
            y0[x] = u0[x] = v0[x] = samples;
        }
        y0 += pitch_y;
        u0 += pitch_uv;
        v0 += pitch_uv;
    }
}


static void __stdcall link_y_to_uv_420(PVideoFrame& dst)
{
    int pitch_y = dst->GetPitch(PLANAR_Y) * 2;
    int pitch_uv = dst->GetPitch(PLANAR_U);
    int width = dst->GetRowSize(PLANAR_U);
    int height = dst->GetHeight(PLANAR_U);
    const uint8_t* y0 = dst->GetReadPtr(PLANAR_Y);
    const uint8_t* y1 = y0 + pitch_y / 2;
    uint8_t* u0 = dst->GetWritePtr(PLANAR_U);
    uint8_t* v0 = dst->GetWritePtr(PLANAR_V);

    if (pitch_y < 64) {
        return;
    }
    __m128i zero = _mm_setzero_si128();
    __m128i all1 = _mm_cmpeq_epi32(zero, zero);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            __m128i xmm0 = _mm_load_si128((__m128i*)(y0 + 2 * x));
            __m128i xmm1 = _mm_load_si128((__m128i*)(y0 + 2 * x + 16));
            __m128i xmm2 = _mm_load_si128((__m128i*)(y1 + 2 * x));
            __m128i xmm3 = _mm_load_si128((__m128i*)(y1 + 2 * x + 16));
            xmm0 = _mm_packs_epi16(_mm_or_si128(xmm0, xmm2), _mm_or_si128(xmm1, xmm3));
            xmm0 = _mm_xor_si128(_mm_cmpeq_epi8(xmm0, zero), all1);
            xmm1 = _mm_load_si128((__m128i*)(u0 + x));
            _mm_store_si128((__m128i*)(u0 + x), _mm_or_si128(xmm0, xmm1));
            xmm1 = _mm_load_si128((__m128i*)(v0 + x));
            _mm_store_si128((__m128i*)(v0 + x), _mm_or_si128(xmm0, xmm1));
        }
        y0 += pitch_y;
        y1 += pitch_y;
        u0 += pitch_uv;
        v0 += pitch_uv;
    }
}


static void __stdcall link_all_420(PVideoFrame& dst)
{
    int pitch_y = dst->GetPitch(PLANAR_Y) * 2;
    int pitch_uv = dst->GetPitch(PLANAR_U);
    int width = dst->GetRowSize(PLANAR_U);
    int height = dst->GetHeight(PLANAR_U);
    uint8_t* y0 = dst->GetWritePtr(PLANAR_Y);
    uint8_t* y1 = y0 + pitch_y / 2;
    uint8_t* u0 = dst->GetWritePtr(PLANAR_U);
    uint8_t* v0 = dst->GetWritePtr(PLANAR_V);

    if (pitch_y < 64) {
        return;
    }
    __m128i zero = _mm_setzero_si128();
    __m128i all1 = _mm_cmpeq_epi32(zero, zero);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            __m128i xmm0, xmm1, val0, val1, val2, val3;
            xmm0 = _mm_load_si128((__m128i*)(u0 + x));
            xmm1 = _mm_load_si128((__m128i*)(v0 + x));
            val0 = _mm_or_si128(xmm0, xmm1);
            val1 = _mm_unpackhi_epi8(val0, val0);
            val0 = _mm_unpacklo_epi8(val0, val0);

            xmm0 = _mm_load_si128((__m128i*)(y0 + 2 * x));
            xmm1 = _mm_load_si128((__m128i*)(y0 + 2 * x + 16));
            val2 = _mm_or_si128(val0, xmm0);
            val3 = _mm_or_si128(val1, xmm1);
            _mm_store_si128((__m128i*)(y0 + 2 * x), val2);
            _mm_store_si128((__m128i*)(y0 + 2 * x + 16), val3);

            xmm0 = _mm_load_si128((__m128i*)(y1 + 2 * x));
            xmm1 = _mm_load_si128((__m128i*)(y1 + 2 * x + 16));
            val2 = _mm_or_si128(val2, xmm0);
            val3 = _mm_or_si128(val3, xmm1);
            _mm_store_si128((__m128i*)(y1 + 2 * x), _mm_or_si128(val0, xmm0));
            _mm_store_si128((__m128i*)(y1 + 2 * x + 16), _mm_or_si128(val1, xmm1));

            val0 = _mm_packs_epi16(val2, val3);
            val0 = _mm_xor_si128(all1, _mm_cmpeq_epi8(val0, zero));
            _mm_store_si128((__m128i*)(u0 + x), val0);
            _mm_store_si128((__m128i*)(v0 + x), val0);
        }
        y0 += pitch_y;
        y1 += pitch_y;
        u0 += pitch_uv;
        v0 += pitch_uv;
    }
}


static void __stdcall link_y_to_uv_422(PVideoFrame& dst)
{
    int pitch_y = dst->GetPitch(PLANAR_Y);
    int pitch_uv = dst->GetPitch(PLANAR_U);
    int width = dst->GetRowSize(PLANAR_U);
    int height = dst->GetHeight(PLANAR_Y);
    const uint8_t* y0 = dst->GetReadPtr(PLANAR_Y);
    uint8_t* u0 = dst->GetWritePtr(PLANAR_U);
    uint8_t* v0 = dst->GetWritePtr(PLANAR_V);

    if (pitch_y < 32) {
        return;
    }
    __m128i zero = _mm_setzero_si128();
    __m128i all1 = _mm_cmpeq_epi32(zero, zero);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            __m128i xmm0 = _mm_load_si128((__m128i*)(y0 + 2 * x));
            __m128i xmm1 = _mm_load_si128((__m128i*)(y0 + 2 * x + 16));
            xmm0 = _mm_packs_epi16(xmm0, xmm1);
            xmm0 = _mm_xor_si128(_mm_cmpeq_epi8(xmm0, zero), all1);
            xmm1 = _mm_load_si128((__m128i*)(u0 + x));
            _mm_store_si128((__m128i*)(u0 + x), _mm_or_si128(xmm0, xmm1));
            xmm1 = _mm_load_si128((__m128i*)(v0 + x));
            _mm_store_si128((__m128i*)(v0 + x), _mm_or_si128(xmm0, xmm1));
        }
        y0 += pitch_y;
        u0 += pitch_uv;
        v0 += pitch_uv;
    }
}


static void __stdcall link_all_422(PVideoFrame& dst)
{
    int pitch_y = dst->GetPitch(PLANAR_Y);
    int pitch_uv = dst->GetPitch(PLANAR_U);
    int width = dst->GetRowSize(PLANAR_U);
    int height = dst->GetHeight(PLANAR_Y);
    uint8_t* y0 = dst->GetWritePtr(PLANAR_Y);
    uint8_t* u0 = dst->GetWritePtr(PLANAR_U);
    uint8_t* v0 = dst->GetWritePtr(PLANAR_V);

    if (pitch_y < 32) {
        return;
    }
    __m128i zero = _mm_setzero_si128();
    __m128i all1 = _mm_cmpeq_epi32(zero, zero);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            __m128i xmmy0, xmmy1, xmmu, xmmv, val0, val1;
            xmmy0 = _mm_load_si128((__m128i*)(y0 + 2 * x));
            xmmy1 = _mm_load_si128((__m128i*)(y0 + 2 * x + 16));
            xmmu  = _mm_load_si128((__m128i*)(u0 + x));
            xmmv  = _mm_load_si128((__m128i*)(v0 + x));
            val0 = _mm_or_si128(xmmu, xmmv);
            val1 = _mm_unpackhi_epi8(val0, val0);
            val0 = _mm_unpacklo_epi8(val0, val0);
            val0 = _mm_or_si128(xmmy0, val0);
            val1 = _mm_or_si128(xmmy1, val1);
            _mm_store_si128((__m128i*)(y0 + 2 * x), val0);
            _mm_store_si128((__m128i*)(y0 + 2 * x + 16), val1);
            val0 = _mm_packs_epi16(val0, val1);
            val0 = _mm_xor_si128(all1, _mm_cmpeq_epi8(val0, zero));
            _mm_store_si128((__m128i*)(u0 + x), val0);
            _mm_store_si128((__m128i*)(v0 + x), val0);
        }
        y0 += pitch_y;
        u0 += pitch_uv;
        v0 += pitch_uv;
    }
}


static void __stdcall link_y_to_uv_411(PVideoFrame& dst)
{
    int pitch_y = dst->GetPitch(PLANAR_Y);
    int pitch_uv = dst->GetPitch(PLANAR_U);
    int width = dst->GetRowSize(PLANAR_U);
    int height = dst->GetHeight(PLANAR_Y);
    const uint8_t* y0 = dst->GetReadPtr(PLANAR_Y);
    uint8_t* u0 = dst->GetWritePtr(PLANAR_U);
    uint8_t* v0 = dst->GetWritePtr(PLANAR_V);

    if (pitch_y < 64) {
        return;
    }
    __m128i zero = _mm_setzero_si128();
    __m128i all1 = _mm_cmpeq_epi32(zero, zero);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            __m128i xmm0 = _mm_load_si128((__m128i*)(y0 + 4 * x));
            __m128i xmm1 = _mm_load_si128((__m128i*)(y0 + 4 * x + 16));
            __m128i xmm2 = _mm_load_si128((__m128i*)(y0 + 4 * x + 32));
            __m128i xmm3 = _mm_load_si128((__m128i*)(y0 + 4 * x + 48));
            xmm0 = _mm_packs_epi32(xmm0, xmm1);
            xmm1 = _mm_packs_epi32(xmm2, xmm3);
            xmm0 = _mm_packs_epi16(xmm0, xmm1);
            xmm0 = _mm_xor_si128(_mm_cmpeq_epi8(xmm0, zero), all1);
            xmm1 = _mm_load_si128((__m128i*)(u0 + x));
            _mm_store_si128((__m128i*)(u0 + x), _mm_or_si128(xmm0, xmm1));
            xmm1 = _mm_load_si128((__m128i*)(v0 + x));
            _mm_store_si128((__m128i*)(v0 + x), _mm_or_si128(xmm0, xmm1));
        }
        y0 += pitch_y;
        u0 += pitch_uv;
        v0 += pitch_uv;
    }
}


static void __stdcall link_all_411(PVideoFrame& dst)
{
    int pitch_y = dst->GetPitch(PLANAR_Y);
    int pitch_uv = dst->GetPitch(PLANAR_U);
    int width = dst->GetRowSize(PLANAR_U);
    int height = dst->GetHeight(PLANAR_U);
    uint8_t* y0 = dst->GetWritePtr(PLANAR_Y);
    uint8_t* u0 = dst->GetWritePtr(PLANAR_U);
    uint8_t* v0 = dst->GetWritePtr(PLANAR_V);

    if (pitch_y < 64) {
        return;
    }
    __m128i zero = _mm_setzero_si128();
    __m128i all1 = _mm_cmpeq_epi32(zero, zero);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            __m128i xmm0, xmm1, val0, val1, val2, val3;
            xmm0 = _mm_load_si128((__m128i*)(u0 + x));
            xmm1 = _mm_load_si128((__m128i*)(v0 + x));
            val0 = _mm_or_si128(xmm0, xmm1);
            val2 = _mm_unpackhi_epi8(val0, val0);
            val0 = _mm_unpacklo_epi8(val0, val0);
            val1 = _mm_unpackhi_epi16(val0, val0);
            val0 = _mm_unpacklo_epi16(val0, val0);
            val3 = _mm_unpackhi_epi16(val2, val2);
            val2 = _mm_unpacklo_epi16(val2, val2);

            xmm0 = _mm_load_si128((__m128i*)(y0 + 4 * x));
            val0 = _mm_or_si128(val0, xmm0);
            _mm_store_si128((__m128i*)(y0 + 4 * x), val0);

            xmm0 = _mm_load_si128((__m128i*)(y0 + 4 * x + 16));
            val1 = _mm_or_si128(val1, xmm0);
            _mm_store_si128((__m128i*)(y0 + 4 * x + 16), val1);

            xmm0 = _mm_load_si128((__m128i*)(y0 + 4 * x + 32));
            val2 = _mm_or_si128(val2, xmm0);
            _mm_store_si128((__m128i*)(y0 + 4 * x + 326), val2);

            xmm0 = _mm_load_si128((__m128i*)(y0 + 4 * x + 48));
            val3 = _mm_or_si128(val3, xmm0);
            _mm_store_si128((__m128i*)(y0 + 4 * x + 48), val3);

            val0 = _mm_packs_epi32(val0, val1);
            val1 = _mm_packs_epi32(val2, val3);
            val0 = _mm_packs_epi16(val0, val1);
            val0 = _mm_xor_si128(all1, _mm_cmpeq_epi8(val0, zero));
            _mm_store_si128((__m128i*)(u0 + x), val0);
            _mm_store_si128((__m128i*)(v0 + x), val0);
        }
        y0 += pitch_y;
        u0 += pitch_uv;
        v0 += pitch_uv;
    }
}


const link_planes_func link_y_to_uv[] = {
    link_y_to_uv_444, link_y_to_uv_422, link_y_to_uv_420, link_y_to_uv_411
};

const link_planes_func link_all[] = {
    link_all_444, link_all_422, link_all_420, link_all_411
};
