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
    target_link_libraries(libexternal INTERFACE n2k)

    # target_include_directories(libexternal INTERFACE pirate)
    add_subdirectory(pirate)
    target_link_libraries(libexternal INTERFACE pirate)

    # Treat vendored ksgpu/n2k headers as system headers for consumers, so their pre-existing
    # diagnostics (e.g. -Wclass-conversion in ksgpu/Array.hpp) don't fail kotekan's -Werror build.
    # Their own sources still see -Wall.
    foreach(tgt ksgpu n2k pirate)
        get_target_property(inc ${tgt} INTERFACE_INCLUDE_DIRECTORIES)
        if(inc)
            set_property(
                TARGET ${tgt}
                APPEND
                PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
        endif()
    endforeach()
endif()
