#!/bin/sh

# clang format version
CLANG_FORMAT=clang-format-8

# kotekan root directory
KOTEKAN_DIR="."

# Flag to enable iwyu (default OFF)
ENABLE_IWYU="OFF"

# number of jobs for iwyu
N_JOBS=4

# exit if one test fails
EXIT_ON_FAILURE="ON"

usage() {
  echo "Usage: $0 [ -d KOTEKAN_DIR ] [ -i ENABLE_IWYU ] [ -j NUM_JOBS ] [-e ENABLE_EXIT_ON_FAILURE]
        Run all the linting tools to make sure the code passes kotekan's CI checks.

        This includes:
        - black:              python code formatting (black.readthedocs.io)
        - clang-format:       C/C++ code formatting (clang.llvm.org/docs/ClangFormat.html)
        - iwyu (optional):    include-what-you-use for C/C++. Make sure to run cmake with
                              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON first
                              (include-what-you-use.org)

        -d KOTEKAN_DIR        Path to kotekan root directory
        -i ENABLE_IWYU        \"ON\" or \"OFF\" to enable or disable include-what-you-use (default:
                              \"OFF\")
        -j NUM_JOBS           Number of concurrent jobs for iwyu (Default: 4)
        -e ENABLE_EXIT_ON_FAILURE
                              \"ON\" or \"OFF\" to enable or disable  exiting if a test fails
                              (default: \"ON\")
" 1>&2
}
exit_abnormal() {
  usage
  exit 1
}

echo "lint.sh: So you don't have to push kotekan twice."

# parse command line arguments
while getopts ":d:i:j:e:" options; do
  case "${options}" in
    d)
      KOTEKAN_DIR=${OPTARG}
      ;;
    j)
      N_JOBS=${OPTARG}
      ;;
    e)
      EXIT_ON_FAILURE=${OPTARG}
      if ! [ $EXIT_ON_FAILURE = "ON" ] && ! [ $EXIT_ON_FAILURE = "OFF" ] ; then
        echo "Error: ENABLE_EXIT_ON_FAILURE must be a ON or OFF (was $EXIT_ON_FAILURE)."
        exit_abnormal
        exit 1
      fi
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

if ! [ $EXIT_ON_FAILURE = "OFF" ]; then
    # exit when any command fails
    set -e
fi

# include-what-you-use
if ! [ $ENABLE_IWYU = "OFF" ]; then
    #CXX=clang++ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
    echo "Running iwyu. If it fails make sure to run cmake with
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON first.\nThis could take a while..."
    iwyu_tool -j $N_JOBS -p . -- -Xiwyu --no_fwd_decls -Xiwyu --mapping_file=${KOTEKAN_DIR}/iwyu.kotekan.imp -Xiwyu --max_line_length=100 | tee iwyu.out
    echo "Applying suggested changes..."
    python2 /usr/bin/fix_include --nosafe_headers --comments < iwyu.out
else
    echo "fast mode enabled, skipping IWYU (add option -i ON to disable fast mode)"
fi

# clang-format
echo "Running clang-format..."
find $KOTEKAN_DIR -type d -name "build" -prune -o -type d -name "external" -prune -o -regex '.*\.\(cpp\|hpp\|c\|h\)' -exec $CLANG_FORMAT -style=file -i {} \;
git diff --exit-code

# black
echo "Running black..."
black --exclude "/(\.eggs|\.git|\.hg|\.mypy_cache|\.nox|\.tox|\.venv|\.svn|_build|buck-out|build|dist|docs)/" $KOTEKAN_DIR
git diff --exit-code

exit 0
