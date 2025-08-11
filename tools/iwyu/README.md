The include-what-you-use mapping files are from the
[iwyu repository](https://github.com/include-what-you-use/include-what-you-use) version
`0.22`. To grab updated files, you can run a command like
```
ls *.imp | xargs -I{} wget https://raw.githubusercontent.com/include-what-you-use/include-what-you-use/refs/tags/0.22/{}
```
The file kotekan CI uses to reference all mapping files is `iwyu.kotekan.imp` in the
root directory of the kotekan repository.
It can be applied by running
```
export CXX=clang++
export CC=clang
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIWYU=ON ..
iwyu_tool -p . -- -Xiwyu --no_fwd_decls -Xiwyu --max_line_length=100 -Xiwyu --mapping_file=/full/path/to/iwyu.kotekan.imp | tee iwyu.out
python3 ../tools/iwyu/fix_includes.py --nosafe_headers --comments
make
```
