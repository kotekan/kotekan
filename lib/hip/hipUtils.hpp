/**
 * @file
 * @brief Helper function for using HIP
 *  - CHECK_HIP_ERROR
 *  - CHECK_HIP_ERROR_NON_OO
 */

#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include "errors.h"
#include "kotekanLogging.hpp"

#include <assert.h>

// Convenience function for checking HIP runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
#define CHECK_HIP_ERROR(result)                                                                    \
    if (result != hipSuccess) {                                                                    \
        FATAL_ERROR_NON_OO("Error at {:s}:{:d}; Error type: {:s}", __FILE__, __LINE__,             \
                           hipGetErrorString(result));                                             \
    }

#define CHECK_HIP_ERROR_NON_OO(result)                                                             \
    if (result != hipSuccess) {                                                                    \
        FATAL_ERROR_NON_OO("Error at {:s}:{:d}; Error type: {:s}", __FILE__, __LINE__,             \
                           hipGetErrorString(result));                                             \
    }

#endif // HIP_UTILS_H
