project(libexternal)
add_library(libexternal INTERFACE)

target_include_directories(libexternal INTERFACE fmt)
target_include_directories(libexternal INTERFACE gsl)
target_include_directories(libexternal INTERFACE json)

add_subdirectory(MurmurHash3)
target_link_libraries(libexternal INTERFACE MurmurHash3)

add_subdirectory(modp_b64)
target_link_libraries(libexternal INTERFACE modp_b64)

if(${USE_CUDA})
    add_subdirectory(gputils)
    target_link_libraries(libexternal INTERFACE gputils)
    add_subdirectory(n2k)
    target_link_libraries(libexternal INTERFACE n2k)
endif()
