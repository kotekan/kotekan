install(
    FILES kotekan.service kotekan-kko.service
    DESTINATION /etc/systemd/system/
    PERMISSIONS OWNER_WRITE GROUP_WRITE)
