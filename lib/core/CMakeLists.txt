cmake_minimum_required( VERSION 2.8 )
project( kotekan_core )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

set (KOTEKAN_CORE_SOURCES
              basebandReadoutManager.cpp
              basebandApiManager.cpp
              buffer.c
              bufferContainer.cpp
              bufferFactory.cpp
              Config.cpp
              configUpdater.cpp
              errors.c
              kotekanLogging.cpp
              kotekanMode.cpp
              KotekanProcess.cpp
              metadata.c
              metadataFactory.cpp
              processFactory.cpp
              prometheusMetrics.cpp
              restServer.cpp
       )

add_library ( kotekan_core
              ${KOTEKAN_CORE_SOURCES})

target_link_libraries( kotekan_core ${CMAKE_THREAD_LIBS_INIT} )
#target_link_libraries( kotekan_core libinclude )
#target_link_libraries( kotekan_core event )
#target_link_libraries( kotekan_core event_pthreads )

if( OPENSSL_FOUND )
    target_link_libraries(kotekan_core ${OPENSSL_LIBRARIES})
endif()
