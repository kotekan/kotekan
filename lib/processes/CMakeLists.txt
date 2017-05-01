cmake_minimum_required( VERSION 2.8 )
project( kotekan_libs )

include_directories (${CHIME_SOURCE_DIR}/lib)

add_library ( kotekan_processes
              beamformingPostProcess.cpp
              chrxUplink.cpp
              computeDualpolPower.cpp
              dpdkWrapper.cpp
              fullPacketDump.cpp
              gpuPostProcess.cpp
              nDiskFileWrite.cpp
              networkPowerStream.cpp
              nullProcess.cpp
              pyPlotResult.cpp
              rawFileRead.cpp
              rawFileWrite.cpp
              vdifStream.cpp
              )

# This seems like a circular reference, seems like there should be a
# way to avoid this.
target_link_libraries( kotekan_processes kotekan_libs )
