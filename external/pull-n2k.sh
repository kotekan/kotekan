#!/bin/bash
set -e
set -x
cd "$(dirname "$0")" # change to directory containing this script (kotekan/external or something)

temp_dir=$(mktemp -d)
src_dir=$(pwd)

# Output the path of the temporary directory
echo "Temporary directory created: $temp_dir"
echo "If this script fails, you should delete this directory."

cd "$temp_dir"

# Clone n2k, check out right branch / commit
git clone -b master --single-branch https://github.com/kmsmith137/n2k.git
cd n2k
# Copy over needed files
cp template_instantiations/make-instantiation.py "$src_dir/n2k/template_instantiations/make-instantiation.py"
cp src_lib/precompute_offsets.cu "$src_dir/n2k/src_lib/precompute_offsets.cu"
cp src_lib/kernel_table.cu "$src_dir/n2k/src_lib/kernel_table.cu"
cp src_lib/Correlator.cu "$src_dir/n2k/src_lib/Correlator.cu"
cp include/n2k/Correlator.hpp "$src_dir/n2k/include/n2k/Correlator.hpp"
cp include/n2k/CorrelatorKernel.hpp "$src_dir/n2k/include/n2k/CorrelatorKernel.hpp"
cd ..

# Clone gputils, check out right branch / commit
git clone -b master --single-branch https://github.com/kmsmith137/gputils.git
cd gputils
# Copy over needed files
cp -R include/* "$src_dir/gputils/include/."
cp -R src_lib/* "$src_dir/gputils/src_lib/."

cd ../..
rm -rf tmp
