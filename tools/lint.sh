#!/bin/sh

# Run all the linting tools to make sure the code passes kotekan's CI checks.
#
# This includes:
# - black:          python code formatting (black.readthedocs.io)
# - clang-format:   C/C++ code formatting (clang.llvm.org/docs/ClangFormat.html)
# - iwyu:           include-what-you-use for C/C++ (include-what-you-use.org)

if [ "build" != "${PWD##*/}" ]; then
    echo "Expected current working dir: build (was ${PWD##*/}). Please create and or change to the
    build directory in your clone of the kotekan repository."
fi

# include-what-you-use
CXX=clang++ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
iwyu_tool -p . -- --mapping_file=${WORKSPACE}/iwyu.kotekan.imp --max_line_length=100 > iwyu.out
python2 /usr/bin/fix_include --comments < iwyu.out

# clang-format
make clang-format

# black
black --exclude docs ..
