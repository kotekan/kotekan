include(GNUInstallDirs)

install(
    FILES kotekan.service
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/systemd/system"
    PERMISSIONS OWNER_WRITE GROUP_WRITE)
