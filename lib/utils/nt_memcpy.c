#include "nt_memcpy.h"

#include <assert.h> // for assert
#ifdef __AVX__
#include <emmintrin.h> // for __m128i, _mm_stream_si128, _mm_load_si128, _mm_loadu_si128
#else
#include <memory.h>
#endif
#include <stdint.h> // for uintptr_t
#include <stdlib.h> // for size_t
#ifdef __AVX__
#include <xmmintrin.h> // for _MM_HINT_NTA, _mm_prefetch
#endif

#ifdef __AVX__
// Assumes that the dest pointer is 16 byte alligned.
// Assumes that the len is divisible by 128
void nt_memcpy(void* dest, void* src, size_t len) {
    // Assumes dest is 16 byte alligned
    // NOTE: This must be true or else non-temporal writes are not possible.
    assert((((uintptr_t)dest) & 0xF) == 0);
    assert((len % 128) == 0);

    if ((((uintptr_t)src) & 0xF) == 0) {
        nt_aligned_memcpy(dest, src, len);
    } else {
        nt_unaligned_memcpy(dest, src, len);
    }
}

// Assumes that the source is 16 byte alligned.
void nt_aligned_memcpy(void* dest, void* src, size_t len) {
    size_t num_loops = len >> 7;

    // To make the math work nicer, we cast to 128-bit array
    __m128i* src_p = (__m128i*)src;
    __m128i* dest_p = (__m128i*)dest;

    for (size_t i = 0; i < num_loops; ++i) {
        _mm_prefetch(((const char*)src) + 768, _MM_HINT_NTA);
        _mm_prefetch(((const char*)src) + 832, _MM_HINT_NTA);
        // Unroll for 8 128 bit registers
        __m128i xmm_reg0 = _mm_load_si128(src_p);
        __m128i xmm_reg1 = _mm_load_si128(src_p + 1);
        __m128i xmm_reg2 = _mm_load_si128(src_p + 2);
        __m128i xmm_reg3 = _mm_load_si128(src_p + 3);
        __m128i xmm_reg4 = _mm_load_si128(src_p + 4);
        __m128i xmm_reg5 = _mm_load_si128(src_p + 5);
        __m128i xmm_reg6 = _mm_load_si128(src_p + 6);
        __m128i xmm_reg7 = _mm_load_si128(src_p + 7);
        _mm_stream_si128(dest_p, xmm_reg0);
        _mm_stream_si128(dest_p + 1, xmm_reg1);
        _mm_stream_si128(dest_p + 2, xmm_reg2);
        _mm_stream_si128(dest_p + 3, xmm_reg3);
        _mm_stream_si128(dest_p + 4, xmm_reg4);
        _mm_stream_si128(dest_p + 5, xmm_reg5);
        _mm_stream_si128(dest_p + 6, xmm_reg6);
        _mm_stream_si128(dest_p + 7, xmm_reg7);
        src_p += 8;
        dest_p += 8;
    }
}

// Assumes that the source is not 16 byte alligned.
void nt_unaligned_memcpy(void* dest, void* src, size_t len) {
    size_t num_loops = len >> 7;

    // To make the math work nicer, we cast to 128-bit array
    __m128i* src_p = (__m128i*)src;
    __m128i* dest_p = (__m128i*)dest;

    for (size_t i = 0; i < num_loops; ++i) {
        _mm_prefetch(((const char*)src) + 768, _MM_HINT_NTA);
        _mm_prefetch(((const char*)src) + 832, _MM_HINT_NTA);
        // Unroll for 8 128 bit registers
        __m128i xmm_reg0 = _mm_loadu_si128(src_p);
        __m128i xmm_reg1 = _mm_loadu_si128(src_p + 1);
        __m128i xmm_reg2 = _mm_loadu_si128(src_p + 2);
        __m128i xmm_reg3 = _mm_loadu_si128(src_p + 3);
        __m128i xmm_reg4 = _mm_loadu_si128(src_p + 4);
        __m128i xmm_reg5 = _mm_loadu_si128(src_p + 5);
        __m128i xmm_reg6 = _mm_loadu_si128(src_p + 6);
        __m128i xmm_reg7 = _mm_loadu_si128(src_p + 7);
        _mm_stream_si128(dest_p, xmm_reg0);
        _mm_stream_si128(dest_p + 1, xmm_reg1);
        _mm_stream_si128(dest_p + 2, xmm_reg2);
        _mm_stream_si128(dest_p + 3, xmm_reg3);
        _mm_stream_si128(dest_p + 4, xmm_reg4);
        _mm_stream_si128(dest_p + 5, xmm_reg5);
        _mm_stream_si128(dest_p + 6, xmm_reg6);
        _mm_stream_si128(dest_p + 7, xmm_reg7);
        src_p += 8;
        dest_p += 8;
    }
}
#else
inline void nt_memcpy(void* dest, void* src, size_t len) {
    memcpy(dest, src, len);
}
inline void nt_aligned_memcpy(void* dest, void* src, size_t len) {
    memcpy(dest, src, len);
}
inline void nt_unaligned_memcpy(void* dest, void* src, size_t len) {
    memcpy(dest, src, len);
}
#endif
