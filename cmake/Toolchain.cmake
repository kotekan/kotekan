# Common toolchain, standards, warnings, and build flags

include_guard(GLOBAL)

# Require at least c++17 support from the compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set C standard to gnu99
set(CMAKE_C_STANDARD 99)

# OpenCL version macro (quiets a build warning)
add_definitions(-DCL_TARGET_OPENCL_VERSION=220)

# ccache
set(CCACHE_ENABLED OFF)
set(CCACHE_REASON "disabled")
if(${CCACHE})
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "Using ccache from ${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CCACHE_ENABLED ON)
        set(CCACHE_REASON "enabled (${CCACHE_PROGRAM})")
    else()
        message(STATUS "Unable to find ccache")
        set(CCACHE_ENABLED OFF)
        set(CCACHE_REASON "not found")
    endif()
endif()

# Warnings-as-errors
set(WERROR_ENABLED ${WERROR})
if(${WERROR})
    message(STATUS "Treating all warnings as errors")
    if(${CMAKE_CUDA_COMPILER})
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--warn-no-error>)
    endif()
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-Werror>)
endif()

# General warnings
add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-Wall>)
add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-Wextra>)

# Optional override/suggest warnings (only when not using IWYU)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Winconsistent-missing-override HAVE_INCONSISTENT_MISSING_OVERRIDE)
if(HAVE_INCONSISTENT_MISSING_OVERRIDE AND NOT ${IWYU})
    add_compile_options(
        $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-Winconsistent-missing-override>)
endif()
check_cxx_compiler_flag(-Wsuggest-override HAVE_SUGGEST_OVERRIDE)
if(HAVE_SUGGEST_OVERRIDE AND NOT ${IWYU})
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wsuggest-override>)
endif()

# Debug/opt flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -ggdb -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -ggdb -O2")

if(${SUPERDEBUG})
    kmsg_status("Superdebugging enabled!!")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer")
endif()

if(${SANITIZE})
    kmsg_status("Sanitization enabled!!")
    # Break long sanitizer flag lines to satisfy cmakelint line-length rule
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-optimize-sibling-calls -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fno-optimize-sibling-calls -fsanitize=address")
endif()

# IWYU must be set before any targets are added
set(IWYU_ENABLED OFF)
if(IWYU)
    find_program(IWYU_PATH NAMES include-what-you-use iwyu)
    if(NOT IWYU_PATH)
        message(FATAL_ERROR "Could not find the program include-what-you-use")
    endif()
    if(NOT IWYU_MAPPING_FILE)
        set(IWYU_MAPPING_FILE "${KOTEKAN_SOURCE_DIR}/iwyu.kotekan.imp")
    endif()
    message(
        STATUS "IWYU enabled: Using iwyu from ${IWYU_PATH} and mapping file ${IWYU_MAPPING_FILE}")
    execute_process(
        COMMAND ${IWYU_PATH} --version
        OUTPUT_VARIABLE IWYU_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "IWYU version: ${IWYU_VERSION}")
    set(IWYU_PATH_AND_OPTIONS
        ${IWYU_PATH}
        -Xiwyu
        --max_line_length=100
        -Xiwyu
        --mapping_file=${IWYU_MAPPING_FILE}
        -Xiwyu
        --no_fwd_decls)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU_PATH_AND_OPTIONS})
    set(CMAKE_C_INCLUDE_WHAT_YOU_USE ${IWYU_PATH_AND_OPTIONS})
    set(IWYU_ENABLED ON)
else()
    set(IWYU_ENABLED OFF)
endif()

# Default build type
set(DEFAULT_BUILD_TYPE "Debug")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "Test")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}")
endif()

# Debug logging defines for Debug and Test
if(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES Test)
    add_definitions(-DDEBUGGING)
    kmsg_status("DEBUG logging enabled")
    kmsg_status("Asserts enabled")
endif()

# General defines and tuning
add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-D_GNU_SOURCE>)
if(NOT DEFINED ARCH)
    set(ARCH "native")
endif()
add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-march=${ARCH}>)
add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-mtune=${ARCH}>)
add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-I/opt/rocm/include>)

# OpenMP flags
if(${USE_OMP})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Linker defaults
if(APPLE)
    # MacOS: turn off ASLR for better debugging
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm -Wl,-no_pie")
else()
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++ -L/opt/rocm/lib -lm")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")

# Quiet noisy cmake policy triggered by some FindXYZ scripts
if(NOT ${CMAKE_VERSION} VERSION_LESS "3.12.0")
    cmake_policy(SET CMP0075 NEW)
endif()

# MacOS specifics
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions(-DMAC_OSX)
endif()

# Disable complex math NaN/INF range checking for performance
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fcx-limited-range HAVE_CX_LIMITED_RANGE)
if(HAVE_CX_LIMITED_RANGE AND NOT ${IWYU})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcx-limited-range")
endif()
check_c_compiler_flag(-fcx-limited-range HAVE_C_LIMITED_RANGE)
if(HAVE_C_LIMITED_RANGE AND NOT ${IWYU})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcx-limited-range")
endif()
