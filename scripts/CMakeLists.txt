install(
    FILES kotekan.service kotekan-pco.service
    DESTINATION /etc/systemd/system/
    PERMISSIONS OWNER_WRITE GROUP_WRITE)
