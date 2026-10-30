include(GNUInstallDirs)

# Install the entire config tree (YAMLs, Jinja templates, examples, tests) to
# <PREFIX>/etc/kotekan/ (typically).
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_SYSCONFDIR}/kotekan"
    PATTERN "CMakeLists.txt" EXCLUDE)
