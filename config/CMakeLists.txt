include(GNUInstallDirs)

# Install the entire config tree (YAMLs, Jinja templates, examples, tests) to /etc/kotekan/.
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_SYSCONFDIR}/kotekan"
    PATTERN "CMakeLists.txt" EXCLUDE)
