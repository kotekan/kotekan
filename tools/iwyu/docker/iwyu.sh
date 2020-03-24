#!/bin/sh

iwyu_tool -j 2 -p . -- -Xiwyu --no_fwd_decls -Xiwyu --mapping_file=/code/iwyu.kotekan.imp -Xiwyu --max_line_length=100 | tee iwyu.out
python2 /usr/bin/fix_include --nosafe_headers --comments < iwyu.out
