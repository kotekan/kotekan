#!/bin/bash
# Note: Don't use `set -e` here. We want to scan all files,
# collect any issues, and exit non-zero at the end if needed.

path="."

# get path argument
if [ "$#" = 1 ]; then
    path="$1"
fi

echo "Checking all cmake files in '$path' and its subdirectories."

# Track tool failures, format changes, and lint warnings.
had_issues=0

for tool in cmake-format cmake-lint; do
    if ! command -v "$tool" > /dev/null; then
        echo "Error: required tool '$tool' was not found in PATH." >&2
        had_issues=1
    fi
done
if [[ $had_issues -ne 0 ]]; then
    exit 1
fi

# Run cmakelint on all CMakeList.txt recursively.
shopt -s globstar
for file in "$path"/{,**/}CMakeLists.txt; do
    # Format in-place, then check if that file changed
    if ! cmake-format -c "$path"/tools/cmake_format_config.py -i -- "$file"; then
        echo "cmake-format failed for: $file" >&2
        had_issues=1
    fi
    if ! git diff --exit-code -- "$file" > /dev/null; then
        echo "cmake-format applied changes to: $file" >&2
        had_issues=1
    fi

    # Capture cmake-lint output; treat any output as an issue
    if ! lint_out=$(cmake-lint --suppress-decorations -c "$path"/tools/cmake_format_config.py -- "$file" 2>&1); then
        echo "cmake-lint failed for: $file" >&2
        had_issues=1
    fi
    if [[ -n "$lint_out" ]]; then
        echo "$lint_out"
        had_issues=1
    fi
done

# Run cmakelint on all .cmake files recursively.
shopt -s nullglob
for file in "$path"/cmake/*.cmake; do
    # For some reason the FindHIP.cmake script from AMD breaks the cmakelint parser
    if [[ $file =~ "FindHIP" ]]; then
        continue
    fi
    # Format in-place, then check if that file changed
    if ! cmake-format -c "$path"/tools/cmake_format_config.py -i -- "$file"; then
        echo "cmake-format failed for: $file" >&2
        had_issues=1
    fi
    if ! git diff --exit-code -- "$file" > /dev/null; then
        echo "cmake-format applied changes to: $file" >&2
        had_issues=1
    fi

    # Capture cmake-lint output; treat any output as an issue
    if ! lint_out=$(cmake-lint --suppress-decorations -c "$path"/tools/cmake_format_config.py -- "$file" 2>&1); then
        echo "cmake-lint failed for: $file" >&2
        had_issues=1
    fi
    if [[ -n "$lint_out" ]]; then
        echo "$lint_out"
        had_issues=1
    fi
done

echo "Done."

# Exit non-zero if any issues were found
if [[ $had_issues -ne 0 ]]; then
    exit 1
fi
