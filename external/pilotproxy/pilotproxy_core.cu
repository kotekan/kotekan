// Builds the vendored core for one detector window length. CMakeLists.txt compiles this file once
// per length, setting PILOTPROXY_DETECTOR_WINDOW_SAMPLES and PILOTPROXY_NAMESPACE and renaming the
// core's C functions, so that both builds link into one binary.

#include "pilotproxy_core.hpp"

// The core's standard headers, included here so that they stay outside the namespace.
#include <climits>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PILOTPROXY_NAMESPACE {

#include "pilotproxy_detector.cu"

namespace reference {
#include "pilotproxy_reference.cpp"
} // namespace reference

extern const PilotProxyCore core = {
    PilotProxy_GetSpecs,
    PilotProxy_GetFineSpecs,
    PilotProxy_LastError,
    PilotProxy_Create,
    PilotProxy_Destroy,
    PilotProxy_SetStream,
    PilotProxy_Compute_RowSums_I32,
    PilotProxy_Compute_FusedFine_U64,
    PilotProxy_Compute_FusedFineMask_U64,
    PilotProxy_Compute_FusedFineMaskWithSupport_U64,
    reference::pilotproxy_ref_row_sums_i32,
    reference::pilotproxy_ref_fine_powers_u64,
    reference::pilotproxy_ref_powers_from_fine_u64,
};

} // namespace PILOTPROXY_NAMESPACE
