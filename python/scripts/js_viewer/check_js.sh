#!/bin/sh
# Syntax-check every viewer JS file. Run after ANY viewer edit -- the app is served straight
# to the browser with no build step, so a syntax error only shows up as a silently dead panel.
#
# The files are ES modules but named .js, and `node --check` parses .js as CommonJS: pointing it
# straight at them reports "To load an ES module..." for every file, which looks like 35 failures
# and is really zero. Copying to a .mjs makes node parse them as the modules they are.
#
# Needs: apt-get install --no-install-recommends nodejs   (2026-07-24: 0 upgraded, no nvidia/
# kernel packages touched -- safe under the GX10 fragile-driver rule).
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
