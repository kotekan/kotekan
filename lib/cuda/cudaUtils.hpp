/**
 * @file
 * @brief Helper function for using CUDA
 *  - CHECK_CUDA_ERROR
 *  - CHECK_CUDA_ERROR_NON_OO
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "errors.h"
#include "kotekanLogging.hpp"

#include <cstdlib>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
#define CHECK_CUDA_ERROR(cmd)                                                                      \
    do {                                                                                           \
        const cudaError_t result = (cmd);                                                          \
        if (result != cudaSuccess) {                                                               \
            internal_logging(LOG_ERR, __log_prefix, "Error at {:s}:{:d}; Error type: {:s} ({:d})", \
                             __FILE__, __LINE__, cudaGetErrorString(result), int(result));         \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

// Similar to CHECK_CUDA_ERROR, but for cu API functions (starting with "cu", vs starting with
// "cuda").
#define CHECK_CU_ERROR(cmd)                                                                        \
    do {                                                                                           \
        const CUresult result = (cmd);                                                             \
        if (result != CUDA_SUCCESS) {                                                              \
            const char* errstr = NULL;                                                             \
            cuGetErrorString(result, &errstr);                                                     \
            internal_logging(LOG_ERR, __log_prefix, "Error at {:s}:{:d}; Error type: {:s}",        \
                             __FILE__, __LINE__, errstr);                                          \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

#define CHECK_CUDA_ERROR_NON_OO(cmd)                                                               \
    do {                                                                                           \
        const cudaError_t result = (cmd);                                                          \
        if (result != cudaSuccess) {                                                               \
            kotekan::kotekanLogging::internal_logging(                                             \
                LOG_ERR, "", "Error at {:s}:{:d}; Error type: {:s}", __FILE__, __LINE__,           \
                cudaGetErrorString(result));                                                       \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

#endif // CUDA_UTILS_H
