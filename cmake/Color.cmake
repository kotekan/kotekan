# Colorized messaging helpers

if(NOT DEFINED KOTEKAN_COLOR_LOADED)
    set(KOTEKAN_COLOR_LOADED TRUE)

    string(ASCII 27 KTK_ESC)
    set(KTK_RESET "${KTK_ESC}[0m")
    set(KTK_GREEN "${KTK_ESC}[1;32m")
    set(KTK_YELLOW "${KTK_ESC}[1;33m")
    set(KTK_RED "${KTK_ESC}[1;31m")

    # Prior (non-summary) messages: keep uncolored and non-warning for readability kmsg_ok: Print an
    # OK/info message during configure
    function(kmsg_ok msg)
        message(STATUS "${msg}")
    endfunction()

    # kmsg_warn: Print a warning-style message during configure
    function(kmsg_warn msg)
        message(WARNING "${KTK_YELLOW}${msg}${KTK_RESET}")
    endfunction()

    # kmsg_error: Print an error-style message during configure
    function(kmsg_error msg)
        message(SEND_ERROR "${KTK_RED}${msg}${KTK_RESET}")
    endfunction()

    # kmsg_status: Print a standard STATUS line (used by the feature summary)
    function(kmsg_status msg)
        message(STATUS "${msg}")
    endfunction()
endif()
