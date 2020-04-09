project( MurmurHash3 )

add_library ( MurmurHash3 MurmurHash3.cpp )
target_include_directories( MurmurHash3 INTERFACE . )
