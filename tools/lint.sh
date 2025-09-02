#!/bin/bash

# clang format version
CLANG_FORMAT=clang-format-18
if ! command -v "$CLANG_FORMAT" > /dev/null 2>&1; then
    echo "Error: clang-format command '$CLANG_FORMAT' not found" >&2
    exit 1
fi

# kotekan root directory (infer from this script location)
KOTEKAN_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"

# Prefer system-wide kotekan Python env for Python tooling (e.g., black)
# Detect a usable black command, prioritizing /opt/kotekan_env when present.
BLACK_CMD=""
if [ -x /opt/kotekan_env/bin/black ]; then
  BLACK_CMD="/opt/kotekan_env/bin/black"
elif command -v black >/dev/null 2>&1; then
  BLACK_CMD="$(command -v black)"
elif [ -x /opt/kotekan_env/bin/python ] && /opt/kotekan_env/bin/python -m black --version >/dev/null 2>&1; then
  BLACK_CMD="/opt/kotekan_env/bin/python -m black"
fi

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
        - cmakelint           lint CMakeList files

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
    CXX=clang++
    CC=clang
    echo "Running iwyu. If it fails make sure cmake compiles with
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON first.\nThis could take a while..."
    mkdir -p ${KOTEKAN_DIR}/build-iwyu
    (cd ${KOTEKAN_DIR}/build-iwyu && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIWYU=ON ..)
    (cd ${KOTEKAN_DIR}/build-iwyu && iwyu_tool -j $N_JOBS -p . -- -Xiwyu --no_fwd_decls -Xiwyu --max_line_length=100 -Xiwyu --mapping_file=${KOTEKAN_DIR}/iwyu.kotekan.imp | tee iwyu.out)
    echo "Applying suggested changes..."
    python3 ${KOTEKAN_DIR}/tools/iwyu/fix_includes.py --nosafe_headers --comments < ${KOTEKAN_DIR}/build-iwyu/iwyu.out
else
    echo "fast mode enabled, skipping IWYU (add option -i ON to disable fast mode)"
fi

# clang-format
echo "Running clang-format..."
find $KOTEKAN_DIR -type d \( -name "build-iwyu" -name "build" -o -name "external" \) -prune -o -type f -regex '.*\.\(cpp\|hpp\|c\|h\)' -exec $CLANG_FORMAT -style=file -i {} \;
if ! git diff --exit-code; then
    echo "Error: clang-format found formatting issues" >&2
    ERROR=1
fi

# black
echo "Running black..."
if [ -z "$BLACK_CMD" ]; then
    echo "Error: could not find a usable 'black'. If available, consider using /opt/kotekan_env." >&2
    exit 1
fi
echo "Using black at: $BLACK_CMD"
$BLACK_CMD --exclude="/(\.eggs|\.git|\.hg|\.mypy_cache|\.nox|\.tox|\.venv|\.svn|_build|buck-out|build|build-iwyu|dist|docs|external|tools/iwyu)/" $KOTEKAN_DIR
if ! git diff --exit-code; then
    echo "Error: black found formatting issues" >&2
    ERROR=1
fi

# cmake-list
echo "Running cmakelint..."
if ! source ${KOTEKAN_DIR}/tools/cmakelint.sh ${KOTEKAN_DIR}; then
    echo "Error: cmakelint failed" >&2
    ERROR=1
fi

if [[ ${ERROR} -ne 0 ]]; then
    echo "One or more formatting checks failed" >&2
    exit 1
fi

exit 0
