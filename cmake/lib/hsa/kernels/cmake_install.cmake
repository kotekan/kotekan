# Install script for directory: /root/arun/kotekan/lib/hsa/kernels

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/var/lib/kotekan/hsa_kernels//")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/var/lib/kotekan/hsa_kernels/" TYPE DIRECTORY FILES "")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/var/lib/kotekan/hsa_kernels/N2.hsaco;/var/lib/kotekan/hsa_kernels/presum.hsaco;/var/lib/kotekan/hsa_kernels/null.hsaco;/var/lib/kotekan/hsa_kernels/pulsar_beamformer.hsaco;/var/lib/kotekan/hsa_kernels/reorder.hsaco;/var/lib/kotekan/hsa_kernels/rfi_chime.hsaco;/var/lib/kotekan/hsa_kernels/rfi_vdif.hsaco;/var/lib/kotekan/hsa_kernels/transpose.hsaco;/var/lib/kotekan/hsa_kernels/unpack_shift_beamform_flip.hsaco;/var/lib/kotekan/hsa_kernels/unpack_shift_beamform_noflip.hsaco;/var/lib/kotekan/hsa_kernels/upchannelize_flip.hsaco;/var/lib/kotekan/hsa_kernels/upchannelize_noflip.hsaco;/var/lib/kotekan/hsa_kernels/pulsar_beamformer_float.hsaco")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/var/lib/kotekan/hsa_kernels" TYPE FILE FILES
    "/root/arun/kotekan/cmake/lib/hsa/kernels/N2.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/presum.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/null.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/pulsar_beamformer.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/reorder.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/rfi_chime.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/rfi_vdif.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/transpose.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/unpack_shift_beamform_flip.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/unpack_shift_beamform_noflip.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/upchannelize_flip.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/upchannelize_noflip.hsaco"
    "/root/arun/kotekan/cmake/lib/hsa/kernels/pulsar_beamformer_float.hsaco"
    )
endif()

