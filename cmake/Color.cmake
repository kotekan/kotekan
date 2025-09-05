# Colorized messaging helpers

if(NOT DEFINED KOTEKAN_COLOR_LOADED)
    set(KOTEKAN_COLOR_LOADED TRUE)

    if(NOT WIN32)
        string(ASCII 27 KTK_ESC)
        set(KTK_RESET "${KTK_ESC}[0m")
        set(KTK_GREEN "${KTK_ESC}[1;32m")
        set(KTK_YELLOW "${KTK_ESC}[1;33m")
        set(KTK_RED "${KTK_ESC}[1;31m")
    else()
        set(KTK_RESET "")
        set(KTK_GREEN "")
        set(KTK_YELLOW "")
    endif()

    # Prior (non-summary) messages: keep uncolored and non-warning for readability
    function(kmsg_ok MSG)
        # Intentionally silent before summary to avoid redundancy
    endfunction()

    function(kmsg_warn MSG)
        # Intentionally silent before summary to avoid redundancy
    endfunction()

    function(kmsg_status MSG)
        message(STATUS "${MSG}")
    endfunction()
endif()
