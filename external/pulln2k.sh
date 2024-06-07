#!/bin/bash
set -e
set -x
cd "$(dirname "$0")"

temp_dir=$(mktemp -d)
src_dir=$(pwd)

# Output the path of the temporary directory
echo "Temporary directory created: $temp_dir"
echo "If this script fails, you should delete this directory."

cd "$temp_dir"

git clone -b rfimask --single-branch https://github.com/kmsmith137/n2k.git
cd n2k
git checkout c42aac7
cp template_instantiations/make-instantiation.py "$src_dir/n2k/template_instantiations/make-instantiation.py"
cp src/precompute_offsets.cu "$src_dir/n2k/src/precompute_offsets.cu"
cp src/kernel_table.cu "$src_dir/n2k/src/kernel_table.cu"
cp src/Correlator.cu "$src_dir/n2k/src/Correlator.cu"
cp include/n2k_kernel.hpp "$src_dir/n2k/include/n2k_kernel.hpp"
cp include/n2k.hpp "$src_dir/n2k/include/n2k.hpp"
cd ..

git clone -b master --single-branch https://github.com/kmsmith137/gputils.git
cd gputils
git checkout 017f016
cp -R include/* "$src_dir/gputils/include/."
cp -R src_lib/* "$src_dir/gputils/src_lib/."

cd ../..
rm -rf tmp
