# From <https://raw.githubusercontent.com/xtensor-stack/xtensor-julia-cookiecutter/master/%7B%7Bcookiecutter.github_project_name%7D%7D/cmake/FindJulia.cmake>

if(Julia_FOUND)
    return()
endif()

####################
# Julia Executable #
####################

find_program(Julia_EXECUTABLE julia DOC "Julia executable")
MESSAGE(STATUS "Julia_EXECUTABLE:     ${Julia_EXECUTABLE}")

#################
# Julia Version #
#################

execute_process(
    COMMAND ${Julia_EXECUTABLE} --version
    OUTPUT_VARIABLE Julia_VERSION_STRING
)

string(
    REGEX REPLACE ".*([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1"
      Julia_VERSION_STRING ${Julia_VERSION_STRING}
)

MESSAGE(STATUS "Julia_VERSION_STRING: ${Julia_VERSION_STRING}")

####################
# Julia Share Path #
####################

execute_process(
    COMMAND ${Julia_EXECUTABLE} --eval "print(joinpath(Sys.BINDIR, Base.DATAROOTDIR, \"julia\"))"
    OUTPUT_VARIABLE Julia_SHARE_PATH
)

#############################
# Julia Include Directories #
#############################

execute_process(
    COMMAND ${Julia_SHARE_PATH}/julia-config.jl --cflags
    OUTPUT_VARIABLE Julia_INCLUDE_DIRS
)

string(REGEX REPLACE "\n" "" Julia_INCLUDE_DIRS ${Julia_INCLUDE_DIRS})
string(REGEX REPLACE "'" "" Julia_INCLUDE_DIRS ${Julia_INCLUDE_DIRS})
string(REGEX REPLACE " " "  " Julia_INCLUDE_DIRS ${Julia_INCLUDE_DIRS})
string(CONCAT Julia_INCLUDE_DIRS " " ${Julia_INCLUDE_DIRS} " ")
string(REGEX REPLACE " -[^I][^ ]* " " " Julia_INCLUDE_DIRS ${Julia_INCLUDE_DIRS})
string(REGEX REPLACE " -I" " " Julia_INCLUDE_DIRS ${Julia_INCLUDE_DIRS})
string(REGEX REPLACE "  " " " Julia_INCLUDE_DIRS ${Julia_INCLUDE_DIRS})
string(REGEX REPLACE "^ " "" Julia_INCLUDE_DIRS ${Julia_INCLUDE_DIRS})
string(REGEX REPLACE " $" "" Julia_INCLUDE_DIRS ${Julia_INCLUDE_DIRS})

set(Julia_INCLUDE_DIRS ${Julia_INCLUDE_DIRS}
    CACHE PATH "Location of Julia include files")
MESSAGE(STATUS "Julia_INCLUDE_DIRS:   ${Julia_INCLUDE_DIRS}")

#################
# Julia Library #
#################

execute_process(
    COMMAND ${Julia_SHARE_PATH}/julia-config.jl --ldflags
    OUTPUT_VARIABLE Julia_LIBRARY_DIR
)

string(REGEX REPLACE "\n" "" Julia_LIBRARY_DIR ${Julia_LIBRARY_DIR})
string(REGEX REPLACE "'" "" Julia_LIBRARY_DIR ${Julia_LIBRARY_DIR})
string(REGEX REPLACE " " "  " Julia_LIBRARY_DIR ${Julia_LIBRARY_DIR})
string(CONCAT Julia_LIBRARY_DIR " " ${Julia_LIBRARY_DIR} " ")
string(REGEX REPLACE " -[^L][^ ]* " " " Julia_LIBRARY_DIR ${Julia_LIBRARY_DIR})
string(REGEX REPLACE " -L" " " Julia_LIBRARY_DIR ${Julia_LIBRARY_DIR})
string(REGEX REPLACE "  " " " Julia_LIBRARY_DIR ${Julia_LIBRARY_DIR})
string(REGEX REPLACE "^ " "" Julia_LIBRARY_DIR ${Julia_LIBRARY_DIR})
string(REGEX REPLACE " $" "" Julia_LIBRARY_DIR ${Julia_LIBRARY_DIR})

execute_process(
    COMMAND ${Julia_SHARE_PATH}/julia-config.jl --ldlibs
    OUTPUT_VARIABLE Julia_LIBRARY_NAME
)

string(REGEX REPLACE "\n" "" Julia_LIBRARY_NAME ${Julia_LIBRARY_NAME})
string(REGEX REPLACE "'" "" Julia_LIBRARY_NAME ${Julia_LIBRARY_NAME})
string(REGEX REPLACE " " "  " Julia_LIBRARY_NAME ${Julia_LIBRARY_NAME})
string(CONCAT Julia_LIBRARY_NAME " " ${Julia_LIBRARY_NAME} " ")
string(REGEX REPLACE " -[^l][^ ]* " " " Julia_LIBRARY_NAME ${Julia_LIBRARY_NAME})
string(REGEX REPLACE " -l" " " Julia_LIBRARY_NAME ${Julia_LIBRARY_NAME})
string(REGEX REPLACE "  " " " Julia_LIBRARY_NAME ${Julia_LIBRARY_NAME})
string(REGEX REPLACE "^ " "" Julia_LIBRARY_NAME ${Julia_LIBRARY_NAME})
string(REGEX REPLACE " $" "" Julia_LIBRARY_NAME ${Julia_LIBRARY_NAME})

find_library(Julia_LIBRARY ${Julia_LIBRARY_NAME} HINTS ${Julia_LIBRARY_DIR} REQUIRED)

set(Julia_LIBRARY ${Julia_LIBRARY}
    CACHE PATH "Julia library")
MESSAGE(STATUS "Julia_LIBRARY:        ${Julia_LIBRARY}")

##################################
# Check for Existence of Headers #
##################################

find_path(Julia_MAIN_HEADER julia.h HINTS ${Julia_INCLUDE_DIRS})

###########################
# FindPackage Boilerplate #
###########################

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Julia
    REQUIRED_VARS   Julia_LIBRARY Julia_INCLUDE_DIRS
    VERSION_VAR     Julia_VERSION_STRING
    FAIL_MESSAGE    "Julia not found"
)
