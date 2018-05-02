#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
//#include <x86intrin.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <omp.h>


// Summary of error codes.
//


// Function type for transposing blocks.
typedef void (*TBFunDef)(void* A, void* B, const size_t lda, const size_t ldb,
        const size_t block_size, const size_t elem_size);


// Function type for transposing edge rows and columns.
typedef void (*TEFunDef)(void* A, void* B, const size_t lda, const size_t ldb,
        const size_t sda, const size_t sdb, const size_t elem_size);


// Simple (unblocked) transpose for arrays of n-byte words.
int simple_transpose(void* in, void* out, const size_t lda, const size_t ldb,
        const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    #pragma omp parallel for
    for(size_t i = 0; i < lda; i++) {
        for(size_t j = 0; j < ldb; j++) {
            memcpy(&B[(j*lda + i) * elem_size],
                   &A[(i*ldb + j) * elem_size], elem_size);
        }
    }
    return 0;
}


// General function for transposing blocks and edge rows/columns. The
// transpose within a block is performed by the passed function.
int blocked_transpose_fun(TBFunDef tbfun, void* in, void* out,
        const size_t lda, const size_t ldb, const size_t block_size,
        const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    // OMP Scheduling optimization.
    // This is designed such that threads are never simultaniousely
    // accessing the same rows or columns of blocks at the same time and
    // thus interfering with each other's cache.
    // For 4 threads this seems to boost the performance by 20%.
    size_t n_thread = omp_get_max_threads();
    size_t n_chunk_min = n_thread * 8;    // For good granularity.
    size_t nba = lda / block_size, nbb = ldb / block_size;
    size_t chunk;
    if (nba > n_chunk_min && nbb > n_chunk_min) {
        chunk = nbb + nbb / n_thread;
    } else if (nba * nbb > n_chunk_min) {
        chunk = nba * nbb / n_chunk_min;
    } else chunk = 1;
    // Main blocked transpose.
    #pragma omp parallel for schedule(static, chunk) collapse(2)
    //#pragma omp parallel for
    for(size_t i = 0; i <= lda - block_size; i += block_size) {
        for(size_t j = 0; j <= ldb - block_size; j += block_size) {
            (*tbfun)(&A[(i*ldb + j) * elem_size],
                     &B[(j*lda + i) * elem_size], 
                     lda, ldb, block_size, elem_size);
        }
    }
    // Transpose the edges that don't fit in blocks.
    size_t bl_edge_a = lda - lda % block_size;
    size_t bl_edge_b = ldb - ldb % block_size;
    #pragma omp parallel for
    for(size_t i = bl_edge_a; i < lda; i++) {
        for(size_t j = 0; j < ldb; j++) {
            memcpy(&B[(j*lda + i) * elem_size],
                   &A[(i*ldb + j) * elem_size], elem_size);
        }
    }
    #pragma omp parallel for
    for(size_t j = bl_edge_b; j < ldb; j++) {
        for(size_t i = 0; i < lda; i++) {
            memcpy(&B[(j*lda + i) * elem_size],
                   &A[(i*ldb + j) * elem_size], elem_size);
        }
    }
    return 0;
}


// --------- Simple blocked transpose functions ---------


// Transpose a block using memcpy for the copy.
void transpose_block_memcpy(void* in, void* out, const size_t lda,
        const size_t ldb, const size_t block_size, const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    for(size_t i = 0; i < block_size; i++) {
        for(size_t j = 0; j < block_size; j++) {
            memcpy(&B[(j*lda + i) * elem_size],
                   &A[(i*ldb + j) * elem_size], elem_size);
        }
    }
}


// Macro for-loop over differently typed arrys.
#define transpose_block_type(in, out, lda, ldb, block_size, type_t) {        \
        type_t* A = (type_t*) in;                                            \
        type_t* B = (type_t*) out;                                           \
        for(size_t i = 0; i < block_size; i++) {                             \
            for(size_t j = 0; j < block_size; j++) {                         \
                B[j*lda + i] = A[i*ldb + j];                                 \
            }                                                                \
        }                                                                    \
    }


// Transpose a block using intrinsic assignment for the copy for common sizes.
void transpose_block(void* in, void* out, const size_t lda, const size_t ldb,
        const size_t block_size, const size_t elem_size) {
    switch(elem_size) {
        case 1:
            transpose_block_type(in, out, lda, ldb, block_size, int8_t);
            break;
        case 2:
            transpose_block_type(in, out, lda, ldb, block_size, int16_t);
            break;
        case 4:
            transpose_block_type(in, out, lda, ldb, block_size, int32_t);
            break;
        case 8:
            transpose_block_type(in, out, lda, ldb, block_size, int64_t);
            break;
        default:
            transpose_block_memcpy(in, out, lda, ldb, block_size, elem_size);
    }
}


// Blocked transpose.
int blocked_transpose(void* in, void* out, const size_t lda, const size_t ldb,
                      const size_t block_size, const size_t elem_size) {
    TBFunDef tbfun = &transpose_block;
    int err = blocked_transpose_fun(tbfun, in, out, lda, ldb, block_size,
            elem_size);
    return err;
}


// Blocked transpose, force the use of memcpy. Slower.
int blocked_transpose_memcpy(void* in, void* out, const size_t lda,
        const size_t ldb, const size_t block_size, const size_t elem_size) {
    TBFunDef tbfun = &transpose_block_memcpy;
    int err = blocked_transpose_fun(tbfun, in, out, lda, ldb, block_size,
            elem_size);
    return err;
}




// ------------ x86 optimized transpose functions. -----------------

inline void transpose_4x4block_SSE_32(float* A, float* B, const size_t lda,
                                   const size_t ldb) {
    __m128 row1 = _mm_load_ps(&A[0*ldb]);
    __m128 row2 = _mm_load_ps(&A[1*ldb]);
    __m128 row3 = _mm_load_ps(&A[2*ldb]);
    __m128 row4 = _mm_load_ps(&A[3*ldb]);
    _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
    _mm_store_ps(&B[0*lda], row1);
    _mm_store_ps(&B[1*lda], row2);
    _mm_store_ps(&B[2*lda], row3);
    _mm_store_ps(&B[3*lda], row4);
}


void transpose_block_SSE_32(void* in, void* out, const size_t lda, const size_t ldb,
                           const size_t block_size, const size_t elem_size) {
    float* A = (float*) in;
    float* B = (float*) out;
    for(size_t i = 0; i < block_size; i += 4) {
        for(size_t j = 0; j < block_size; j += 4) {
            transpose_4x4block_SSE_32(&A[i*ldb +j], &B[j*lda + i], lda, ldb);
        }
    }
}


// Blocked transpose for 32 bit words using SSE instructrions.
int blocked_transpose_SSE_32(void* in, void* out, const size_t lda, const size_t ldb,
                            const size_t block_size) {
    // Using SSE requires the arrays to be byte aligned.
    if ((intptr_t) in % 16 || (intptr_t) out % 16) {
        return 10;
    }
    if (ldb % 4 || lda % 4) {
        return 11;
    }
    if (block_size % 4 || block_size == 0) {
        return 12;
    }
    TBFunDef tbfun = &transpose_block_SSE_32;
    int err = blocked_transpose_fun(tbfun, in, out, lda, ldb, block_size, 4);
    return err;
}


inline void transpose_4x4block_AVX_64(double* A, double* B, const size_t lda,
                                   const size_t ldb) {
    __m256d row0 = _mm256_load_pd(&A[0*ldb]);
    __m256d row1 = _mm256_load_pd(&A[1*ldb]);
    __m256d row2 = _mm256_load_pd(&A[2*ldb]);
    __m256d row3 = _mm256_load_pd(&A[3*ldb]);
    __m256d tmp3, tmp2, tmp1, tmp0;
    tmp0 = _mm256_unpacklo_pd(row0, row1);
    tmp1 = _mm256_unpackhi_pd(row0, row1);
    tmp2 = _mm256_unpacklo_pd(row2, row3);
    tmp3 = _mm256_unpackhi_pd(row2, row3);
    row0 = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
    row1 = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
    row2 = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);
    row3 = _mm256_permute2f128_pd(tmp1, tmp3, 0x31);
    _mm256_store_pd(&B[0*lda], row0);
    _mm256_store_pd(&B[1*lda], row1);
    _mm256_store_pd(&B[2*lda], row2);
    _mm256_store_pd(&B[3*lda], row3);

}

void transpose_block_AVX_64(void* in, void* out, const size_t lda, const size_t ldb,
                           const size_t block_size, const size_t elem_size) {
    double* A = (double*) in;
    double* B = (double*) out;
    for(size_t i = 0; i < block_size; i += 4) {
        for(size_t j = 0; j < block_size; j += 4) {
            transpose_4x4block_AVX_64(&A[i*ldb +j], &B[j*lda + i], lda, ldb);
        }
    }
}


// Blocked transpose for 64 bit words using AVX instructrions.
int blocked_transpose_AVX_64(void* in, void* out, const size_t lda, const size_t ldb,
                            const size_t block_size) {
    // Using SSE requires the arrays to be byte aligned.
    if ((intptr_t) in % 32 || (intptr_t) out % 32) {
        return 10;
    }
    if (ldb % 4 || lda % 4) {
        return 11;
    }
    if (block_size % 4 || block_size == 0) {
        return 12;
    }
    TBFunDef tbfun = &transpose_block_AVX_64;
    int err = blocked_transpose_fun(tbfun, in, out, lda, ldb, block_size, 8);
    return err;
}




