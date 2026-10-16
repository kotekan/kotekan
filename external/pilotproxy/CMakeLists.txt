# Kotekan build glue; VENDOR.json identifies the unmodified detector sources.
project(pilotproxy CXX CUDA)

# Match lib/cuda's default; an explicit cache setting takes precedence.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(PILOTPROXY_DETECTOR_WINDOW_SAMPLES 64 CACHE STRING "PilotProxy samples per window (64 or 128)")
set_property(CACHE PILOTPROXY_DETECTOR_WINDOW_SAMPLES PROPERTY STRINGS 64 128)
if(NOT PILOTPROXY_DETECTOR_WINDOW_SAMPLES MATCHES "^(64|128)$")
    message(FATAL_ERROR "PILOTPROXY_DETECTOR_WINDOW_SAMPLES must be 64 or 128")
endif()

add_library(
    pilotproxy STATIC f_statistic.cu # CUDA kernel + extern "C" API (FStat_*)
                      f_statistic_reference.cpp # bit-accurate CPU reference (fstat_ref_*)
)

# CHORD defaults to 64 samples/window; CHIME uses 128. Each block contains 128 windows. Consumers read this
# private setting through FStat_GetSpecs(). Without NDEBUG the core synchronizes its stream after
# every call, and kotekan's default Test build does not define it.
target_compile_definitions(pilotproxy PRIVATE FSTAT_DETECTOR_WINDOW_SAMPLES=${PILOTPROXY_DETECTOR_WINDOW_SAMPLES} NDEBUG)

# Match the other vendored libraries' system-header treatment.
target_include_directories(pilotproxy SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
