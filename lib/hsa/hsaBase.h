/**
 * @file
 * @brief HSA runtime start and memory allocation functions which must be in C for compatibility
 * - kotekan_hsa_start
 * - hsa_host_malloc
 * - hsa_host_free
 * - kotekan_hsa_stop
 */
#ifndef HSA_KOTEKAN_BASE_H
#define HSA_KOTEKAN_BASE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "errors.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_finalize.h"
#include "hsa/hsa_ext_amd.h"

#include <assert.h>

/**
 * @brief Start the HSA run time, must be called before any GPU related calls are made.
 */
void kotekan_hsa_start();

/**
 * @brief Allocates Host memory (CPU RAM) which can be accessed by a GPU
 *
 * @param len The size in bytes of the memory to allocate
 * @param gpu_id The GPU ID (zero based) which is allowed to access this memory
 * @return void* A pointer to the GPU memory, or NULL if the operation failed.
 */
void * hsa_host_malloc(size_t len, uint32_t gpu_id);

/**
 * @brief Frees memory allocated with hsa_host_malloc
 *
 * @param ptr Pointer to the memory to free.
 */
void hsa_host_free(void *ptr);

/**
 * @brief Cleanly shutdown the HSA runtime
 */
void kotekan_hsa_stop();

#ifdef __cplusplus
}
#endif

#endif
