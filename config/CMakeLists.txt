# Installs all the .yaml files to /etc/kotekan/

FILE(GLOB configFiles RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
                               "${CMAKE_CURRENT_SOURCE_DIR}/*.yaml")

INSTALL(DIRECTORY DESTINATION /etc/kotekan/)
INSTALL(FILES ${configFiles} DESTINATION /etc/kotekan/)