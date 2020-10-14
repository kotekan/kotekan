# ##################################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell cop- ies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM- PLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNE- CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ##################################################################################################

# findHCC does not currently address versioning, i.e. a rich directory structure where version
# number is a subdirectory under root Also, supported only on UNIX 64 bit systems.

if(UNIX)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)

        find_library(
            HSA_LIBRARY
            NAMES hsa-runtime64
            PATHS ENV HSA_PATH /opt/rocm
            PATH_SUFFIXES lib)

        if(NOT DEFINED ENV{HSA_PATH})
            set(ENV{HSA_PATH} /opt/rocm/hsa)
        endif()

        find_program(
            HCC
            NAMES hcc
            PATHS ENV HCC_PATH /opt/rocm/hcc
            PATH_SUFFIXES /bin)

        if(NOT DEFINED ENV{HCC_PATH})
            set(ENV{HCC_PATH} /opt/rocm/hcc)
        endif()

        find_path(
            HCC_INCLUDE_DIR
            NAMES hc.hpp
            PATHS ENV NCC_PATH /opt/rocm/hcc
            PATH_SUFFIXES /include)

        set(HSA_LIBRARIES ${HSA_LIBRARY})
        # set(HCC_LIBRARIES ${AMP_LIBRARY})
        set(HCC_INCLUDE_DIRS ${HCC_INCLUDE_DIR})

        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(
            HCC
            FOUND_VAR HCC_FOUND
            REQUIRED_VARS HSA_LIBRARIES HCC_INCLUDE_DIRS HCC)

        mark_as_advanced(HSA_LIBRARIES HCC_INCLUDE_DIRS)

    else()
        message(SEND_ERROR "HCC is currently supported only on 64 bit UNIX platforms")
    endif()
else()
    message(SEND_ERROR "HCC is currently supported on unix platforms")
endif()
