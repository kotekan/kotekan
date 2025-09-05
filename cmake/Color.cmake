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

    # Prior (non-summary) messages: keep uncolored and non-warning for readability kmsg_ok: Print an
    # OK/info message during configure (currently suppressed)
    function(kmsg_ok msg)
        if(FALSE)
            # no-op
        endif()
    endfunction()

    # kmsg_warn: Print a warning-style message during configure (currently suppressed)
    function(kmsg_warn msg)
        if(FALSE)
            # no-op
        endif()
    endfunction()

    # kmsg_status: Print a standard STATUS line (used by the feature summary)
    function(kmsg_status msg)
        message(STATUS "${msg}")
    endfunction()
endif()
