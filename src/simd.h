#ifndef TEM_MOD_SIMD_H
#define TEM_MOD_SIMD_H

#include <cstdint>
#include <emmintrin.h>

#define SFINLINE static __forceinline


SFINLINE __m128i load(const uint8_t* p)
{
    return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
}

SFINLINE __m128i loadu(const uint8_t* p)
{
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

SFINLINE void store(uint8_t* p, const __m128i& x)
{
    _mm_store_si128(reinterpret_cast<__m128i*>(p), x);
}

SFINLINE __m128i subs_u8(const __m128i& x, const __m128i& y)
{
    return _mm_subs_epu8(x, y);
}

SFINLINE __m128i sub_i16(const __m128i& x, const __m128i& y)
{
    return _mm_sub_epi16(x, y);
}

SFINLINE __m128i mullo_i16(const __m128i& x, const __m128i& y)
{
    return _mm_mullo_epi16(x, y);
}

SFINLINE __m128i madd_i16(const __m128i& x, const __m128i& y)
{
    return _mm_madd_epi16(x, y);
}

SFINLINE __m128i or_reg(const __m128i& x, const __m128i& y)
{
    return _mm_or_si128(x, y);
}

SFINLINE __m128i max_u8(const __m128i& x, const __m128i& y)
{
    return _mm_max_epu8(x, y);
}

SFINLINE __m128i max_i16(const __m128i& x, const __m128i& y)
{
    return _mm_max_epi16(x, y);
}

SFINLINE __m128i min_u8(const __m128i& x, const __m128i& y)
{
    return _mm_min_epu8(x, y);
}

SFINLINE __m128i min_i16(const __m128i& x, const __m128i& y)
{
    return _mm_min_epi16(x, y);
}

SFINLINE __m128i unpacklo_i8(const __m128i& x, const __m128i& y)
{
    return _mm_unpacklo_epi8(x, y);
}

SFINLINE __m128i unpacklo_i16(const __m128i& x, const __m128i& y)
{
    return _mm_unpacklo_epi16(x, y);
}

SFINLINE __m128i unpackhi_i8(const __m128i& x, const __m128i& y)
{
    return _mm_unpackhi_epi8(x, y);
}

SFINLINE __m128i unpackhi_i16(const __m128i& x, const __m128i& y)
{
    return _mm_unpackhi_epi16(x, y);
}

SFINLINE __m128i packus_i16(const __m128i& x, const __m128i& y)
{
    return _mm_packus_epi16(x, y);
}

SFINLINE __m128i packs_i16(const __m128i& x, const __m128i& y)
{
    return _mm_packs_epi16(x, y);
}

SFINLINE __m128i cmpgt_i16(const __m128i& x, const __m128i& y)
{
    return _mm_cmpgt_epi16(x, y);
}

SFINLINE __m128i rshift_i16(const __m128i& x, int n)
{
    return _mm_srli_epi16(x, n);
}

SFINLINE __m128i rshift_i32(const __m128i& x, int n)
{
    return _mm_srli_epi32(x, n);
}

SFINLINE __m128i absdiff_u8(const __m128i& x, const __m128i& y)
{
    return or_reg(subs_u8(x, y), subs_u8(y, x));
}

SFINLINE __m128i abs_i16(const __m128i& x)
{
    return max_i16(sub_i16(_mm_setzero_si128(), x), x);
}

SFINLINE __m128i packus_i32(const __m128i& x, const __m128i& y)
{
    __m128i lo = _mm_shufflelo_epi16(x, _MM_SHUFFLE(3, 1, 2, 0));
    lo = _mm_shufflehi_epi16(lo, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i hi = _mm_shufflelo_epi16(y, _MM_SHUFFLE(2, 0, 3, 1));
    hi = _mm_shufflehi_epi16(hi, _MM_SHUFFLE(2, 0, 3, 1));
    return _mm_shuffle_epi32(or_reg(lo, hi), _MM_SHUFFLE(3, 1, 2, 0));
}

#endif

