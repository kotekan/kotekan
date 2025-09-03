#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

path="."

# get path argument
if [ "$#" = 1 ]; then
    path="$1"
fi

echo "Checking all cmake files in '$path' and its subdirectories."

git config --global --add safe.directory /code/kotekan

# Run cmakelint on all CMakeList.txt recursively.
shopt -s globstar
for file in "$path"/{,**/}CMakeLists.txt; do
    cmake-format -c "$path"/tools/cmake_format_config.py -i -- "$file"
    git diff --exit-code
    cmake-lint --suppress-decorations -c "$path"/tools/cmake_format_config.py -- "$file"
done

# Run cmakelint on all .cmake files recursively.
shopt -s nullglob
for file in "$path"/cmake/*.cmake; do
    # For some reason the FindHIP.cmake script from AMD breaks the cmakelint parser
    if [[ $file =~ "FindHIP" ]]; then
        continue
    fi
    cmake-format -c "$path"/tools/cmake_format_config.py -i -- "$file"
    git diff --exit-code
    cmake-lint --suppress-decorations -c "$path"/tools/cmake_format_config.py -- "$file"
done
