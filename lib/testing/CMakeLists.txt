cmake_minimum_required( VERSION 2.8 )
project( libkotekan_testing )

include ( testing.cmake )

add_library ( kotekan_testing ${KOTEKAN_TESTING_SOURCES} )
