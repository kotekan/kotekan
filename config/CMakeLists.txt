# Install the entire config tree (YAMLs, Jinja templates, examples, tests) to /etc/kotekan/.
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION /etc/kotekan
    PATTERN "CMakeLists.txt" EXCLUDE)
