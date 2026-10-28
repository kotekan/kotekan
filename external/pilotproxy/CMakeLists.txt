# Kotekan build glue; VENDOR.json identifies the unmodified detector sources.
project(pilotproxy CXX CUDA)

# Match lib/cuda's default; an explicit cache setting takes precedence.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# The core fixes its window length at compile time, so build it for 64 (CHORD) and 128 (CHIME)
# samples per window; pilotproxy_core.hpp selects one at run time. Each build renames the core's C
# functions so that both link into one library. Without NDEBUG the core synchronizes its stream
# after every call, and kotekan's default Test build does not define it.
set(PILOTPROXY_C_FUNCTIONS
    PilotProxy_GetSpecs
    PilotProxy_GetFeatures
    PilotProxy_GetOptimizationFeatures
    PilotProxy_GetVersion
    PilotProxy_LastError
    PilotProxy_Create
    PilotProxy_Create_Batch
    PilotProxy_Destroy
    PilotProxy_SetStream
    PilotProxy_Compute_RowSums_I32
    PilotProxy_Supports_RowSums
    PilotProxy_Compute_FinePowers_U64
    PilotProxy_Supports_FinePowers
    PilotProxy_Compute_FusedFine_U64
    PilotProxy_Supports_FusedFine
    PilotProxy_Compute_FusedFineMask_U64
    PilotProxy_Supports_FusedFineMask
    PilotProxy_Compute_FusedFineMaskWithSupport_U64
    PilotProxy_Supports_FusedFineMaskWithSupport
    PilotProxy_GetFineSpecs
    pilotproxy_ref_sign_extend_i4
    pilotproxy_ref_unpack_complex_i4
    pilotproxy_ref_complex_mul_conj
    pilotproxy_ref_row_sums_i32
    pilotproxy_ref_fine_powers_u64
    pilotproxy_ref_powers_from_fine_u64)
foreach(window 64 128)
    set(_target pilotproxy_k${window})
    list(TRANSFORM PILOTPROXY_C_FUNCTIONS REPLACE "^(.+)$" "\\1=${_target}_\\1" OUTPUT_VARIABLE
                                                                                _renames)
    add_library(${_target} OBJECT pilotproxy_core.cu)
    target_compile_definitions(
        ${_target} PRIVATE PILOTPROXY_DETECTOR_WINDOW_SAMPLES=${window}
                           PILOTPROXY_NAMESPACE=${_target} ${_renames} NDEBUG)
endforeach()
add_library(pilotproxy STATIC $<TARGET_OBJECTS:pilotproxy_k64> $<TARGET_OBJECTS:pilotproxy_k128>)

# Match the other vendored libraries' system-header treatment.
target_include_directories(pilotproxy SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
