# Installs all the .yaml files to /etc/kotekan/

install(DIRECTORY DESTINATION /etc/kotekan/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION /etc/kotekan
        FILES_MATCHING
        PATTERN "*.j2"
        PATTERN "*.yaml")
