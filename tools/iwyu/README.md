The include-what-you-use mapping files are from the
[iwyu repository](https://github.com/include-what-you-use/include-what-you-use) version
`0.13`. The file kotekan CI uses to reference all mapping files is `iwyu.kotekan.imp` in the
root directory of this repository.
It can be applied by running
```
cd build
export CXX=clang++
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
iwyu_tool -p . -- --mapping_file=/full/path/to/iwyu.kotekan.imp
make clang-format
```
