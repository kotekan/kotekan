project(kotekan)

add_executable(kotekan kotekan.cpp)

target_compile_definitions(kotekan PRIVATE mssse3 __STDC_LIMIT_MACROS)

target_link_libraries(kotekan PRIVATE libexternal m pthread kotekan_libs)

install(
    TARGETS kotekan
    DESTINATION /usr/local/bin
    COMPONENT binaries)
