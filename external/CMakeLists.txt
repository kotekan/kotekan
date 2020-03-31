project( libexternal )
add_library ( libexternal INTERFACE )

target_include_directories( libexternal INTERFACE fmt)
target_include_directories( libexternal INTERFACE gsl)
target_include_directories( libexternal INTERFACE json)

add_subdirectory( MurmurHash3 )
target_link_libraries( libexternal INTERFACE MurmurHash3 )
