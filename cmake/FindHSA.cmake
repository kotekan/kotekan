# The University of Illinois/NCSA Open Source License (NCSA)
#
# Copyright (c) 2014, Advanced Micro Devices, Inc. All rights reserved.
#
# Developed by:
#
# AMD Research and AMD HSA Software Development
#
# Advanced Micro Devices, Inc.
#
# www.amd.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal with the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions
#   and the following disclaimers.
# * Redistributions in binary form must reproduce the above copyright notice, this list of
#   conditions and the following disclaimers in the documentation and/or other materials provided
#   with the distribution.
# * Neither the names of <Name of Development Group, Name of Institution>, nor the names of its
#   contributors may be used to endorse or promote products derived from this Software without
#   specific prior written permission.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.

if(HSA_RUNTIME_INCLUDE_DIR)
    # The HSA information is already in the cache.
    set(HSA_RUNTIME_FIND_QUIETLY TRUE)
endif(HSA_RUNTIME_INCLUDE_DIR)

# Look for the hsa include file path.

# If the HSA_INCLUDE_DIR variable is set, use it for the HSA_RUNTIME_INCLUDE_DIR variable. Otherwise
# set the value to /opt/hsa/include. Note that this can be set when running cmake by specifying -D
# HSA_INCLUDE_DIR=<directory>.

if(NOT DEFINED HSA_INCLUDE_DIR)
    set(HSA_INCLUDE_DIR "/opt/rocm/include")
endif()

# MESSAGE("HSA_INCLUDE_DIR=${HSA_INCLUDE_DIR}")

find_path(
    HSA_RUNTIME_INCLUDE_DIR
    NAMES hsa/hsa.h
    PATHS ${HSA_INCLUDE_DIR})

# If the HSA_LIBRARY_DIR environment variable is set, use it for the HSA_RUNTIME_LIBRARY_DIR
# variable. Otherwise set the value to /opt/hsa/lib. Note that this can be set when running cmake by
# specifying -D HSA_LIBRARY_DIR=<directory>.

if(NOT DEFINED HSA_LIBRARY_DIR)
    set(HSA_LIBRARY_DIR "/opt/rocm/lib")
endif()

# MESSAGE("HSA_LIBRARY_DIR=${HSA_LIBRARY_DIR}")

# Look for the hsa library and, if found, generate the directory.
if(DEFINED CYGWIN)
    # In CYGWIN set the library name directly to the hsa-runtime64.dll. This is a temporary
    # work-around for cmake limitations, and requires that the HSA_RUNTIME_LIBRARY environment
    # variable is set by the user.
    set(HSA_RUNTIME_LIBRARY "${HSA_LIBRARY_DIR}/hsa-runtime64.dll")
else()
    find_library(
        HSA_RUNTIME_LIBRARY
        NAMES hsa-runtime64
        PATHS ${HSA_LIBRARY_DIR})
endif()

get_filename_component(HSA_RUNTIME_LIBRARY_DIR ${HSA_RUNTIME_LIBRARY} DIRECTORY)

# Handle the QUIETLY and REQUIRED arguments and set HSA_FOUND to TRUE if all listed variables are
# TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HSA "Please install 'hsa-runtime' package" HSA_RUNTIME_LIBRARY
                                  HSA_RUNTIME_INCLUDE_DIR)

if(HSA_FOUND)
    set(HSA_LIBRARIES ${HSA_LIBRARY})
else(HSA_FOUND)
    set(HSA_LIBRARIES)
endif(HSA_FOUND)

mark_as_advanced(HSA_RUNTIME_INCLUDE_DIR)
mark_as_advanced(HSA_RUNTIME_LIBRARY_DIR)
mark_as_advanced(HSA_RUNTIME_LIBRARY)
