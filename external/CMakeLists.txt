project(libexternal)
add_library(libexternal INTERFACE)

target_include_directories(libexternal SYSTEM INTERFACE fmt)
target_include_directories(libexternal SYSTEM INTERFACE gsl)
target_include_directories(libexternal SYSTEM INTERFACE json)

add_subdirectory(MurmurHash3)
target_link_libraries(libexternal INTERFACE MurmurHash3)

add_subdirectory(modp_b64)
target_link_libraries(libexternal INTERFACE modp_b64)

if(CUDAToolkit_FOUND)
    add_subdirectory(ksgpu)
    target_link_libraries(libexternal INTERFACE ksgpu)
    add_subdirectory(n2k)
    target_link_libraries(libexternal INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,n2k>")
    # PilotProxy DTV pilot-tone detector core (vendored; used by lib/cuda/cudaPilotProxyDetector and
    # testPilotProxyDetector).
    add_subdirectory(pilotproxy)
    target_link_libraries(libexternal INTERFACE pilotproxy)

    # target_include_directories(libexternal INTERFACE pirate)
    add_subdirectory(pirate)
    target_link_libraries(libexternal INTERFACE pirate)

    # Treat vendored ksgpu/n2k/pilotproxy headers as system headers for consumers, so their
    # pre-existing diagnostics (e.g. -Wclass-conversion in ksgpu/Array.hpp) don't fail kotekan's
    # -Werror build. Their own sources still see -Wall.
    foreach(tgt ksgpu n2k pirate pilotproxy)
        get_target_property(inc ${tgt} INTERFACE_INCLUDE_DIRECTORIES)
        if(inc)
            set_property(
                TARGET ${tgt}
                APPEND
                PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
        endif()
    endforeach()

    # Release builds add the "-fvisibility=hidden" flag (see kotekan:cmake/Toolchain.cmake) which
    # removes all symbols from the .so file unless they are explicitly marked public; that is not
    # done in some of our vendored code, so override that flag.
    foreach(tgt pirate)
        set_property(
            TARGET ${tgt}
            APPEND
            PROPERTY COMPILE_OPTIONS -fvisibility=default)
    endforeach()

endif()

# pilotproxy_decision.h is header only, for consumers built without CUDA (DtvRfiMask).
add_library(pilotproxy_decision INTERFACE)
target_include_directories(pilotproxy_decision SYSTEM INTERFACE pilotproxy)
