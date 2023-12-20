#ifndef PACKET_COPY_H
#define PACKET_COPY_H

#include <assert.h>
#ifdef __AVX2__
#include <emmintrin.h>
#endif
#include <inttypes.h>
#include <rte_cycles.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "errors.h"

#ifdef __AVX2__

// Copy 8 bytes from one location to another,
// locations should not overlap.
// Output must be 4 bytes aligned.
static inline void rte_mov8_nt(uint8_t* dst, const uint8_t* src) {
    long long int value;

    value = *(const long long int*)src;
    _mm_stream_si64((long long int*)dst, value);
}

// Copy 16 bytes from one location to another,
// locations should not overlap.
// Output must be 4 bytes aligned.
static inline void rte_mov16_nt(uint8_t* dst, const uint8_t* src) {
    __m128i xmm0;

    xmm0 = _mm_loadu_si128((const __m128i*)src);
    _mm_stream_si128((__m128i*)dst, xmm0);
}

// Copy 32 bytes from one location to another,
// locations should not overlap.
// Output must be 4 byte aligned.
static inline void rte_mov32_nt(uint8_t* dst, const uint8_t* src) {
    __m256i ymm0;

    ymm0 = _mm256_loadu_si256((const __m256i*)src);
    _mm256_stream_si256((__m256i*)dst, ymm0);
}

// Copy 64-byte blocks from one location to another,
// locations should not overlap.
// Output must be 4 byte aligned.
static inline void rte_mov64blocks_nt(uint8_t* dst, const uint8_t* src, size_t n) {
    __m256i ymm0, ymm1;

    while (n >= 64) {
        ymm0 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src + 0 * 32));
        n -= 64;
        ymm1 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src + 1 * 32));
        src = (const uint8_t*)src + 64;
        _mm256_stream_si256((__m256i*)((uint8_t*)dst + 0 * 32), ymm0);
        _mm256_stream_si256((__m256i*)((uint8_t*)dst + 1 * 32), ymm1);
        dst = (uint8_t*)dst + 64;
    }
}

// Copy 256-byte blocks from one location to another,
// locations should not overlap.
// Dest must be 4 byte aligned, input source can be unaligned.
static inline void rte_mov256blocks_nt(uint8_t* dst, const uint8_t* src, size_t n) {
    __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;

    while (n >= 256) {
        ymm0 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src + 0 * 32));
        n -= 256;
        ymm1 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src + 1 * 32));
        ymm2 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src + 2 * 32));
        ymm3 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src + 3 * 32));
        ymm4 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src + 4 * 32));
        ymm5 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src + 5 * 32));
        ymm6 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src + 6 * 32));
        ymm7 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src + 7 * 32));
        src = (const uint8_t*)src + 256;
        _mm256_stream_si256((__m256i*)((uint8_t*)dst + 0 * 32), ymm0);
        _mm256_stream_si256((__m256i*)((uint8_t*)dst + 1 * 32), ymm1);
        _mm256_stream_si256((__m256i*)((uint8_t*)dst + 2 * 32), ymm2);
        _mm256_stream_si256((__m256i*)((uint8_t*)dst + 3 * 32), ymm3);
        _mm256_stream_si256((__m256i*)((uint8_t*)dst + 4 * 32), ymm4);
        _mm256_stream_si256((__m256i*)((uint8_t*)dst + 5 * 32), ymm5);
        _mm256_stream_si256((__m256i*)((uint8_t*)dst + 6 * 32), ymm6);
        _mm256_stream_si256((__m256i*)((uint8_t*)dst + 7 * 32), ymm7);
        dst = (uint8_t*)dst + 256;
    }
}

// Output space must be 256 bit aligned, and the copy must be at least 512 bytes,
// and the lenght must be a multiple of 256 bits.
// Requires a CPU with AVX and SSE2 instructions.
static inline void copy_block(struct rte_mbuf** pkt, uint8_t* dest, int len, int* offset) {

    int n, bits, unaligned_n;
    const uint8_t* src;

    int local_offset = *offset;
    // Output must be 256-bit aligned
    // assert(((uintptr_t)dest & 0x1F) == 0);

    // fprintf(stderr, "Copy function\n");

    while (len > 0) {

        // Copy as much as possible in the current packet.
        int last_copy_len = len;
        n = (*pkt)->data_len - local_offset;
        if (n > len) {
            n = len;
        }
        len -= n; // We will copy at least n;
        // assert(len >= 0);
        // INFO_F("copy_block pointer %p", *pkt);
        src = rte_pktmbuf_mtod_offset(*pkt, uint8_t*, local_offset);

        // fprintf(stderr, "pkt_data_len = %d, n = %d, local_offset = %d\n", pkt->data_len, n,
        // local_offset);

        // Realigned the output.
        // fprintf(stderr, "dest before realignment: %p\n", dest);
        if ((unaligned_n = 32 - ((uintptr_t)dest & 0x1F)) != 0) {
            // assert(unaligned_n % 8 == 0);
            n -= unaligned_n;
            while (unaligned_n > 0) {
                rte_mov8_nt(dest, src);
                src = (const uint8_t*)src + 8;
                dest = (uint8_t*)dest + 8;
                unaligned_n -= 8;
            }
        }

        // Copy 256-byte blocks.
        // Use copy block function for better instruction order control,
        // which is important when load is unaligned.
        if (n >= 256) {
            // fprintf(stderr, "dest after: %p\n", dest);
            rte_mov256blocks_nt(dest, src, n);
            bits = n;
            n = n & 255;
            bits -= n;
            src = (const uint8_t*)src + bits;
            dest = (uint8_t*)dest + bits;
        }

        // Copy 64-byte blocks.
        // Use copy block function for better instruction order control,
        // which is important when load is unaligned.
        if (n >= 64) {
            rte_mov64blocks_nt((uint8_t*)dest, (const uint8_t*)src, n);
            bits = n;
            n = n & 63;
            bits -= n;
            src = (const uint8_t*)src + bits;
            dest = (uint8_t*)dest + bits;
        }

        if (n >= 32) {
            rte_mov32_nt((uint8_t*)dest, (const uint8_t*)src);
            n -= 32;
            src = (const uint8_t*)src + 32;
            dest = (uint8_t*)dest + 32;
        }

        if (n >= 16) {
            rte_mov16_nt((uint8_t*)dest, (const uint8_t*)src);
            n -= 16;
            src = (const uint8_t*)src + 16;
            dest = (uint8_t*)dest + 16;
        }

        if (n >= 8) {
            rte_mov8_nt((uint8_t*)dest, (const uint8_t*)src);
            n -= 8;
            src = (const uint8_t*)src + 8;
            dest = (uint8_t*)dest + 8;
        }

        // Since we can cross a packet boundry when trying to load a 64-bit register,
        // we need to combind the results of two 64-bit register reads.
        // Since the output is alligned if the lenght requires we go over a packet
        // boundry the next packet must exist.
        if (likely(n > 0)) {

            // At this point n < 8
            uint64_t result = 0;

            // Get the last 8 bytes of the current frame.
            // first_val = (*(const uint64_t *)(src + n - 8)) << (8 - n)*8;

            uint8_t* char_result = (uint8_t*)&result;

            for (int i = 0; i < n; ++i) {
                char_result[i] = *(src + i);
            }

            // Advance the frame
            *pkt = (*pkt)->next;
            // The preconditions don't allow this to happen.
            // assert(pkt != nullptr);
            // assert(n <= pkt->data_len);

            // Get the first 8 bytes of the next frame.
            // INFO_F("copy_block pointer 2 %p", *pkt);
            src = rte_pktmbuf_mtod(*pkt, uint8_t*);
            // fprintf(stderr, "new src: 0x%016llX\n", );
            // second_val = (*(const uint64_t *)src) >> n*8;
            for (int i = 0; i < (8 - n); ++i) {
                char_result[i + n] = *(src + i);
            }

            // Add the two parts together and store them.
            // result = first_val | second_val;
            _mm_stream_si64((long long int*)dest, result);

            // fprintf(stderr, "n: %d first: 0x%016llX, second: 0x%016llX result: 0x%016llX\n" , n,
            // first_val, second_val, result);

            local_offset = 8 - n;
            len -= local_offset;
            dest = (uint8_t*)dest + 8;

        } else {
            // Advance the frame
            if (len > 0) {
                *pkt = (*pkt)->next;
                local_offset = 0;
                *offset = 0;
                assert(pkt != nullptr);
            } else {
                local_offset = local_offset + last_copy_len;
                break;
            }
        }
    }
    *offset = local_offset;
    // assert(len == 0);
}

#else // Non-AVX

static inline void copy_block(struct rte_mbuf** pkt, uint8_t* dest, int len, int* offset) {
    (void)pkt;
    (void)dest;
    (void)len;
    (void)offset;
    ERROR_NON_OO("copy_block not implemented for systems without AVX2");
}

#endif


#ifdef __cplusplus
}
#endif

#endif
