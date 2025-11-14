#!/bin/bash
# Note: Don't use `set -e` here. We want to scan all files,
# collect any issues, and exit non-zero at the end if needed.

path="."

# get path argument
if [ "$#" = 1 ]; then
    path="$1"
fi

# Check for tools
if ! command -v cmake-format > /dev/null 2>&1; then
    echo "Error: cmake-format not found. Please install it (pip install cmake-format)." >&2
    exit 1
fi

echo "Checking all cmake files in '$path' and its subdirectories."

# Track whether any issues were found (format changes or lint warnings)
had_issues=0

# Run cmakelint on all CMakeList.txt recursively.
shopt -s globstar
for file in "$path"/{,**/}CMakeLists.txt; do
    # Format in-place, then check if that file changed
    cmake-format -c "$path"/tools/cmake_format_config.py -i -- "$file"
    if ! git diff --exit-code -- "$file" > /dev/null; then
        echo "cmake-format applied changes to: $file" >&2
        had_issues=1
    fi

    # Capture cmake-lint output; treat any output as an issue
    # Note: cmake-lint might not be installed if cmake-format was installed via some packages,
    # but pip install cmake-format usually provides both.
    if command -v cmake-lint > /dev/null 2>&1; then
        lint_out=$(cmake-lint --suppress-decorations -c "$path"/tools/cmake_format_config.py -- "$file" || true)
        if [[ -n "$lint_out" ]]; then
            echo "$lint_out"
            had_issues=1
        fi
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
    cmake-format -c "$path"/tools/cmake_format_config.py -i -- "$file"
    if ! git diff --exit-code -- "$file" > /dev/null; then
        echo "cmake-format applied changes to: $file" >&2
        had_issues=1
    fi

    if command -v cmake-lint > /dev/null 2>&1; then
        lint_out=$(cmake-lint --suppress-decorations -c "$path"/tools/cmake_format_config.py -- "$file" || true)
        if [[ -n "$lint_out" ]]; then
            echo "$lint_out"
            had_issues=1
        fi
    fi
done

if [ $had_issues -ne 0 ]; then
    echo "Issues found."
    exit 1
fi

echo "No issues found."
exit 0