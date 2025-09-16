if(NOT DEFINED BLAZE_PATH)
    set(BLAZE_PATH "/usr/local/include")
endif()

find_path(
    BLAZE_INCLUDE_DIR
    NAMES blaze/Blaze.h
    PATHS ${BLAZE_PATH} /usr/local/include /usr/include
    PATH_SUFFIXES blaze)

# Check if include directory was found
if(BLAZE_INCLUDE_DIR)
    set(BLAZE_FOUND TRUE)
else()
    set(BLAZE_FOUND FALSE)
endif()

if(BLAZE_FOUND)
    message(STATUS "Blaze include directory found at ${BLAZE_INCLUDE_DIR}")
    set(BLAZE_PATH "${BLAZE_INCLUDE_DIR}")
else()
    message(STATUS "Blaze include directory not found.")
endif()
