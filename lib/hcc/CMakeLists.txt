project(kotekan_hcc)

# /// AMD hcc options need to add -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hcc to cmake command line

if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
    # set the compile options for hcc
    execute_process(
        COMMAND hcc-config --cxxflags
        OUTPUT_VARIABLE HCC_COMPILE_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_COMPILE_FLAGS} ")

    # set the link options for hcc
    execute_process(
        COMMAND hcc-config --ldflags
        OUTPUT_VARIABLE HCC_LINKER_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${HCC_LINKER_FLAGS} ")
endif()

# /// end AMD hcc options

add_library(kotekan_hcc hccGPUThread.cpp)
target_include_directories(kotekan_hcc PUBLIC .)
