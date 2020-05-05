# Finds libevent include path and libraries Sets the following if libevent is found: LibEvent_FOUND,
# LIBEVENT_INCLUDE_DIR, LIBEVENT_LIBRARIES

include(FindPackageHandleStandardArgs)

set(LIBEVENT_SEARCH_PATHS /usr/include /usr/local/include)

find_path(
    LIBEVENT_INCLUDE_DIR
    NAMES event.h
    PATHS ${LIBEVENT_SEARCH_PATHS}
    PATH_SUFFIXES event2)

# We need event core (timers, buffers), pthreads (thread safe call backs), and extra (http)
find_library(LIBEVENT_BASE NAMES event)
find_library(LIBEVENT_CORE NAMES event_core)
find_library(LIBEVENT_PTHREADS NAMES event_pthreads)
find_library(LIBEVENT_EXTRA NAMES event_extra)

set(LIBEVENT_LIBRARIES ${LIBEVENT_BASE} ${LIBEVENT_CORE} ${LIBEVENT_PTHREADS} ${LIBEVENT_EXTRA})

find_package_handle_standard_args(LIBEVENT DEFAULT_MSG LIBEVENT_LIBRARIES LIBEVENT_INCLUDE_DIR)

mark_as_advanced(LIBEVENT_INCLUDE_DIR LIBEVENT_BASE LIBEVENT_CORE LIBEVENT_PTHREADS LIBEVENT_EXTRA)
