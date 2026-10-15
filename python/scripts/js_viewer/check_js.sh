#!/bin/sh
# Syntax-check every viewer JS file. Run after any viewer edit: the app is served straight to
# the browser with no build step, so a syntax error only shows up as a silently dead panel.
# Exits nonzero if any file fails.
#
# The files are ES modules named .js, and `node --check` parses .js as CommonJS: pointed
# straight at them it reports "To load an ES module..." for every file. Copying each to a .mjs
# makes node parse it as the module it is.
#
# Needs node: apt-get install --no-install-recommends nodejs
set -u
dir=$(dirname "$0")
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT
bad=0
n=0
for f in $(find "$dir" -name '*.js' | sort); do
    n=$((n + 1))
    cp "$f" "$tmp/chk.mjs"
    if ! out=$(node --check "$tmp/chk.mjs" 2>&1); then
        bad=$((bad + 1))
        echo "FAIL $f"
        echo "$out" | grep -v "^$tmp" | head -4
    fi
done
echo "checked $n files, $bad failed"
[ "$bad" -eq 0 ]
