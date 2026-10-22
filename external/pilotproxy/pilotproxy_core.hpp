/**
 * @file pilotproxy_core.hpp
 * @brief Run-time choice between the builds of the vendored PilotProxy core.
 *
 * The core fixes its detector window length at compile time. pilotproxy_core.cu builds it for 64
 * (CHORD) and 128 (CHIME) samples per window. Each member points to the f_statistic.h or
 * f_statistic_reference.h function of the same name (without the FStat_ or fstat_ref_ prefix).
 */
#pragma once

#include <cstdint>        // for int8_t
#include <driver_types.h> // for cudaStream_t

struct PilotProxyCore {
    void (*GetSpecs)(int*, int*, int*, int*);
    void (*GetFineSpecs)(int*, int*, int*);
    const char* (*LastError)();
    void* (*Create)(const std::int8_t*, int);
    void (*Destroy)(void*);
    void (*SetStream)(void*, cudaStream_t);
    void (*Compute_RowSums_I32)(void*, const std::int8_t*, int*);
    void (*Compute_FusedFine_U64)(void*, const std::int8_t*, unsigned long long*,
                                  unsigned long long*, int*);
    void (*Compute_FusedFineMask_U64)(void*, const std::int8_t*, int, int,
                                      const unsigned long long*, int, unsigned long long,
                                      unsigned long long*, int*, unsigned long long*, int*);
    void (*Compute_FusedFineMaskWithSupport_U64)(void*, const std::int8_t*, int, int,
                                                 const unsigned long long*, int,
                                                 unsigned long long, unsigned long long*, int*,
                                                 int*, int*, unsigned long long*, int*);
    void (*row_sums_i32)(const std::int8_t*, const std::int8_t*, int, int*);
    void (*fine_powers_u64)(const int*, int, unsigned long long (*)[256]);
    void (*powers_from_fine_u64)(const unsigned long long (*)[256], unsigned long long*);
};

namespace pilotproxy_k64 {
extern const PilotProxyCore core;
}
namespace pilotproxy_k128 {
extern const PilotProxyCore core;
}

/// The core built for @p window_samples samples per window, or nullptr if there is none.
inline const PilotProxyCore* pilotproxy_core(const int window_samples) {
    switch (window_samples) {
        case 64:
            return &pilotproxy_k64::core;
        case 128:
            return &pilotproxy_k128::core;
        default:
            return nullptr;
    }
}
