
#include "nt_memset.h"

#include <assert.h>    // for assert
#include <immintrin.h> // for _mm256_stream_si256, __m256i, _mm256_set1_epi8
#include <stdint.h>    // for uintptr_t, uint8_t
#include <stdlib.h>    // for size_t

// TODO In theory this will work with AVX and not require AVX2, check this.

#ifdef __AVX__
// Assumes that the dest pointer is 32 byte alligned.
// Assumes that the len is divisible by 256
void nt_memset(void* dest, uint8_t val, size_t len) {

    // Assumes dest is 16 byte alligned
    // NOTE: This must be true or else non-temporal writes are not possible.
    assert((((uintptr_t)dest) & 0xF) == 0);
    assert((len % 256) == 0);

    size_t num_loops = len >> 8;

    // To make the math work nicer, we cast to 128-bit array
    __m256i* dest_p = (__m256i*)dest;

    __m256i ymm0;
    ymm0 = _mm256_set1_epi8((char)val);
    for (size_t i = 0; i < num_loops; ++i) {
        _mm256_stream_si256(dest_p + 0, ymm0);
        _mm256_stream_si256(dest_p + 1, ymm0);
        _mm256_stream_si256(dest_p + 2, ymm0);
        _mm256_stream_si256(dest_p + 3, ymm0);
        _mm256_stream_si256(dest_p + 4, ymm0);
        _mm256_stream_si256(dest_p + 5, ymm0);
        _mm256_stream_si256(dest_p + 6, ymm0);
        _mm256_stream_si256(dest_p + 7, ymm0);
        dest_p += 8;
    }
}
#else
inline void nt_memset(void* dest, uint8_t val, size_t len) {
    memset(dest, val, len);
}
#endif
