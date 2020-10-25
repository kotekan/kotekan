# Installs all the .yaml files to /etc/kotekan/

file(
    GLOB configFiles
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.yaml")

install(DIRECTORY DESTINATION /etc/kotekan/)
install(FILES ${configFiles} DESTINATION /etc/kotekan/)
