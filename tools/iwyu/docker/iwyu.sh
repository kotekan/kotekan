#!/bin/sh

iwyu_tool -j 2 -p . -- -Xiwyu --no_fwd_decls -Xiwyu --mapping_file=/code/kotekan/iwyu.kotekan.imp -Xiwyu --max_line_length=100 | tee iwyu.out
python2 /usr/bin/fix_include --nosafe_headers --comments < iwyu.out

if [ $? -eq 0 ]
then
  echo "All good."
  exit 0
else
  make clang-format
  git diff
  exit 1
fi
