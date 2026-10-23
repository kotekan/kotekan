#!/bin/sh
# Syntax-check every viewer JS file, and the beam-cube viewer's script. Run after any viewer
# edit: the app is served straight to the browser with no build step, so a syntax error only
# shows up as a silently dead panel. Exits nonzero if any file fails, or if it found no viewer
# files to check.
#
# The files are ES modules named .js, and `node --check` parses .js as CommonJS: pointed
# straight at them it reports "To load an ES module..." for every file. Copying each to a .mjs
# makes node parse it as the module it is. beamcube.js is a classic <script>, checked as .cjs.
#
# Needs node: apt-get install --no-install-recommends nodejs
set -u
dir=$(dirname "$0")
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT
bad=0
n=0

check() {  # check <file> <mjs|cjs>
    n=$((n + 1))
    out=
    if ! cp "$1" "$tmp/chk.$2" || ! out=$(node --check "$tmp/chk.$2" 2>&1); then
        bad=$((bad + 1))
        echo "FAIL $1"
        echo "$out" | grep -v "^$tmp" | head -4
    fi
}

for f in $(find "$dir" -name '*.js' | sort); do
    check "$f" mjs
done
n_viewer=$n
check "$dir/../gnss/beam_cube_viewer/beamcube.js" cjs
echo "checked $n files, $bad failed"
if [ "$n_viewer" -eq 0 ]; then
    echo "FAIL no .js files under $dir"
    exit 1
fi
[ "$bad" -eq 0 ]
