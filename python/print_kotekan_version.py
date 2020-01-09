# Prints the kotekan version using versioneer
# The script is called from make_version.cmake.in
from _version import get_versions

print(get_versions()["version"])
