# Kotekan build glue; VENDOR.json identifies the unmodified detector sources.
project(pilotproxy CXX CUDA)

# Match lib/cuda's default; an explicit cache setting takes precedence.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_library(
    pilotproxy STATIC f_statistic.cu # CUDA kernel + extern "C" API (FStat_*)
                      f_statistic_reference.cpp # bit-accurate CPU reference (fstat_ref_*)
)

# CHORD: 64 samples/window and 128 windows per 8192-sample detector block. Consumers read this
# private setting through FStat_GetSpecs(). Without NDEBUG the core synchronizes its stream after
# every call, and kotekan's default Test build does not define it.
target_compile_definitions(pilotproxy PRIVATE FSTAT_DETECTOR_WINDOW_SAMPLES=64 NDEBUG)

# Match the other vendored libraries' system-header treatment.
target_include_directories(pilotproxy SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
