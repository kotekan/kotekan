#!/bin/bash

# clang format version
CLANG_FORMAT=clang-format-18
if ! command -v "$CLANG_FORMAT" > /dev/null 2>&1; then
    echo "Error: clang-format command '$CLANG_FORMAT' not found" >&2
    exit 1
fi

# kotekan root directory (infer from this script location)
KOTEKAN_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"

# Detect black (python formatter)
# We prioritize the command in PATH, then try python module execution
BLACK_CMD=""
if command -v black >/dev/null 2>&1; then
  BLACK_CMD="black"
elif python3 -m black --version >/dev/null 2>&1; then
  BLACK_CMD="python3 -m black"
fi

# Flag to enable iwyu (default OFF)
ENABLE_IWYU="OFF"

# number of jobs for iwyu
N_JOBS=4

# exit if one test fails
EXIT_ON_FAILURE="ON"

# Track if an error has occurred.
ERROR=0

usage() {
  echo "Usage: $0 [ -d KOTEKAN_DIR ] [ -i ENABLE_IWYU ] [ -j NUM_JOBS ] [ -e EXIT_ON_FAILURE ]"
  exit 1
}

while getopts "d:i:j:e:" o; do
  case "${o}" in
    d)
      KOTEKAN_DIR=${OPTARG}
      ;;
    i)
      ENABLE_IWYU=${OPTARG}
      ;;
    j)
      N_JOBS=${OPTARG}
      ;;
    e)
      EXIT_ON_FAILURE=${OPTARG}
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

if [ -z "${KOTEKAN_DIR}" ]; then
    usage
fi

# iwyu
if [ "${ENABLE_IWYU}" == "ON" ]; then
    echo "Running iwyu..."
    # We need to build with IWYU enabled to generate the report
    # Using a temporary build directory to avoid messing up the main build
    mkdir -p ${KOTEKAN_DIR}/build-iwyu
    # Note: We use -k to keep going even if compilation fails, so we get as many IWYU reports as possible
    (cd ${KOTEKAN_DIR}/build-iwyu && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_C_COMPILER=clang-18 ..)
    (cd ${KOTEKAN_DIR}/build-iwyu && iwyu_tool -j $N_JOBS -p . -- -Xiwyu --no_fwd_decls -Xiwyu --max_line_length=100 -Xiwyu --mapping_file=${KOTEKAN_DIR}/tools/iwyu/iwyu.kotekan.imp | tee iwyu.out)
    echo "Applying suggested changes..."
    python3 ${KOTEKAN_DIR}/tools/iwyu/fix_includes.py --reorder --nosafe_headers --update_comments < ${KOTEKAN_DIR}/build-iwyu/iwyu.out
else
    echo "fast mode enabled, skipping IWYU (add option -i ON to disable fast mode)"
fi

# clang-format
echo "Running clang-format..."
find $KOTEKAN_DIR -type d \( -name "build-iwyu" -o -name "build" -o -name "external" \) -prune -o -type f -regex '.*\.\(cpp\|hpp\|c\|h\)' -exec $CLANG_FORMAT -style=file -i {} \;
if ! git diff --exit-code; then
    echo "Error: clang-format found formatting issues" >&2
    ERROR=1
fi

# black
echo "Running black..."
if [ -z "$BLACK_CMD" ]; then
    echo "Error: could not find a usable 'black'. Please install it (apt install black or pip install black)." >&2
    exit 1
fi
echo "Using black at: $BLACK_CMD"
$BLACK_CMD --exclude="/(\.eggs|\.git|\.hg|\.mypy_cache|\.nox|\.tox|\.venv|_build|buck-out|build|dist|external)/" $KOTEKAN_DIR

if ! git diff --exit-code; then
    echo "Error: black found formatting issues" >&2
    ERROR=1
fi

if [ "$EXIT_ON_FAILURE" == "ON" ] && [ $ERROR -ne 0 ]; then
    exit 1
fi

exit 0