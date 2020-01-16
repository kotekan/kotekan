#!/bin/sh

# clang format version
CLANG_FORMAT=clang-format-8

# kotekan root directory
KOTEKAN_DIR="."

# Flag to enable iwyu (default OFF)
ENABLE_IWYU="OFF"

usage() {
  echo "Usage: $0 [ -d KOTEKAN_DIR ] [ -i ENABLE_IWYU ]
        Run all the linting tools to make sure the code passes kotekan's CI checks.

        This includes:
        - black:              python code formatting (black.readthedocs.io)
        - clang-format:       C/C++ code formatting (clang.llvm.org/docs/ClangFormat.html)
        - iwyu (optional):    include-what-you-use for C/C++. Make sure to run cmake with
                              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON first
                              (include-what-you-use.org)" 1>&2
}
exit_abnormal() {
  usage
  exit 1
}

# exit when any command fails
set -e

echo "lint.sh: So you don't have to push kotekan twice."

# parse command line arguments
while getopts ":d:i:" options; do
  case "${options}" in
    d)
      NAME=${OPTARG}
      ;;
    i)
      ENABLE_IWYU=${OPTARG}
      if ! [ $ENABLE_IWYU = "ON" ] && ! [ $ENABLE_IWYU = "OFF" ] ; then
        echo "Error: ENABLE_IWYU must be a ON or OFF (was $ENABLE_IWYU)."
        exit_abnormal
        exit 1
      fi
      ;;
    :)
      echo "Error: -${OPTARG} requires an argument."
      exit_abnormal
      ;;
    *)
      echo "Error: -Unknown option: ${OPTARG}."
      exit_abnormal
      ;;
  esac
done

# include-what-you-use
if ! [ $ENABLE_IWYU = "OFF" ]; then
    CXX=clang++ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
    echo "Running iwyu. If it fails make sure to run cmake with
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON first. This could take a while..."
    iwyu_tool -j 4 -p . -- --mapping_file=${KOTEKAN_DIR}/iwyu.kotekan.imp --max_line_length=100 | tee iwyu.out
    echo "Applying suggested changes..."
    python2 /usr/bin/fix_include --nosafe_headers --comments < iwyu.out
else
    echo "fast mode enabled, skipping IWYU (add option -i ON to disable fast mode)"
fi

# clang-format
echo "Running clang-format..."
find $KOTEKAN_DIR -type d -name "build" -prune -o -type d -name "include" -prune -o -regex '.*\.\(cpp\|hpp\|c\|h\)' -exec $CLANG_FORMAT -style=file -i {} \;

# black
echo "Running black..."
black --exclude docs $KOTEKAN_DIR
git diff --exit-code

exit 0
