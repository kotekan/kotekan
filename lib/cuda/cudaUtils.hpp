#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
#define CHECK_CUDA_ERROR(result)                                                                   \
    if (result != cudaSuccess) {                                                                   \
        FATAL_ERROR(LOG_ERR, "Error at {:s}:{:d}; Error type: {:s}", __FILE__, __LINE__,            \
                         cudaGetErrorString(result));                                              \                          \
    }

#endif // CUDA_UTILS_H
