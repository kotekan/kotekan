#!/bin/bash
# Note: Don't use `set -e` here. We want to scan all files,
# collect any issues, and exit non-zero at the end if needed.

path="."

# get path argument
if [ "$#" = 1 ]; then
    path="$1"
fi

CMAKE_FORMAT_CMD=""
if command -v uv >/dev/null 2>&1 && uv run cmake-format --version >/dev/null 2>&1; then
    CMAKE_FORMAT_CMD="uv run cmake-format"
fi

CMAKE_LINT_CMD=""
if command -v uv >/dev/null 2>&1 && uv run cmake-lint --version >/dev/null 2>&1; then
    CMAKE_LINT_CMD="uv run cmake-lint"
fi

if [ -z "$CMAKE_FORMAT_CMD" ]; then
    echo "Error: cmakelint.sh could not find cmake-format in uv." >&2
    exit 1
fi

if [ -z "$CMAKE_LINT_CMD" ]; then
    echo "Error: cmakelint.sh could not find cmake-lint in uv." >&2
    exit 1
fi

echo "Checking all cmake files in '$path' and its subdirectories."

# Track whether any issues were found (format changes or lint warnings)
had_issues=0

# Run cmakelint on all CMakeList.txt recursively.
shopt -s globstar
for file in "$path"/{,**/}CMakeLists.txt; do
    # Format in-place, then check if that file changed
    $CMAKE_FORMAT_CMD -c "$path"/tools/cmake_format_config.py -i -- "$file"
    if ! git diff --exit-code -- "$file" > /dev/null; then
        echo "$CMAKE_FORMAT_CMD applied changes to: $file" >&2
        had_issues=1
    fi

    # Capture cmake-lint output; treat any output as an issue
    lint_out=$("$CMAKE_LINT_CMD" --suppress-decorations -c "$path"/tools/cmake_format_config.py -- "$file" || true)
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
    $CMAKE_FORMAT_CMD -c "$path"/tools/cmake_format_config.py -i -- "$file"
    if ! git diff --exit-code -- "$file" > /dev/null; then
        echo "$CMAKE_FORMAT_CMD applied changes to: $file" >&2
        had_issues=1
    fi

    # Capture cmake-lint output; treat any output as an issue
    lint_out=$("$CMAKE_LINT_CMD" --suppress-decorations -c "$path"/tools/cmake_format_config.py -- "$file" || true)
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
