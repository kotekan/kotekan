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
    FStat_GetSpecs
    FStat_GetFeatures
    FStat_GetOptimizationFeatures
    FStat_GetVersion
    FStat_LastError
    FStat_Create
    FStat_Create_Batch
    FStat_Destroy
    FStat_SetStream
    FStat_Compute_RowSums_I32
    FStat_Supports_RowSums
    FStat_Compute_FinePowers_U64
    FStat_Supports_FinePowers
    FStat_Compute_FusedFine_U64
    FStat_Supports_FusedFine
    FStat_Compute_FusedFineMask_U64
    FStat_Supports_FusedFineMask
    FStat_Compute_FusedFineMaskWithSupport_U64
    FStat_Supports_FusedFineMaskWithSupport
    FStat_GetFineSpecs
    fstat_ref_sign_extend_i4
    fstat_ref_unpack_complex_i4
    fstat_ref_complex_mul_conj
    fstat_ref_row_sums_i32
    fstat_ref_fine_powers_u64
    fstat_ref_powers_from_fine_u64)
foreach(window 64 128)
    set(build pilotproxy_k${window})
    list(TRANSFORM PILOTPROXY_C_FUNCTIONS REPLACE "^(.+)$" "\\1=${build}_\\1" OUTPUT_VARIABLE renames)
    add_library(${build} OBJECT pilotproxy_core.cu)
    target_compile_definitions(
        ${build} PRIVATE FSTAT_DETECTOR_WINDOW_SAMPLES=${window} PILOTPROXY_NAMESPACE=${build}
                         ${renames} NDEBUG)
endforeach()
add_library(pilotproxy STATIC $<TARGET_OBJECTS:pilotproxy_k64> $<TARGET_OBJECTS:pilotproxy_k128>)

# Match the other vendored libraries' system-header treatment.
target_include_directories(pilotproxy SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
