include(GNUInstallDirs)

install(
    FILES kotekan.service
    DESTINATION "lib/systemd/system"
    PERMISSIONS OWNER_WRITE GROUP_WRITE)
