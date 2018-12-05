install(FILES kotekan.service DESTINATION /etc/systemd/system/ PERMISSIONS OWNER_WRITE GROUP_WRITE)

install(PROGRAMS yaml_to_json.py DESTINATION /usr/local/bin/ COMPONENT scripts)
install(PROGRAMS gps_yaml_to_json.py DESTINATION /usr/local/bin/ COMPONENT scripts)
