#!/bin/bash
# Check that each running component still matches stack_components.sh: argv and environment.
#
# The environment is half the point: a unit can reproduce a command line exactly and still drop
# a variable that is not on it (the broker's OPENBLAS_NUM_THREADS=1). Nothing fails when that
# happens, and an argv comparison alone passes it.
#
#     stack_contract_gate.sh            # every component that is running
#     stack_contract_gate.sh broker     # just one
# A component that is not running is skipped. Exits non-zero if any checked component
# mismatches, or if none was named and none is running: a gate that checked nothing has not
# passed.
set -u
here=$(cd "$(dirname "$0")" && pwd)
bad=0 checked=0

check() {
    local name=$1 unit=$2 chain=${3:-}
    local pid; pid=$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || echo 0)
    [ "${pid:-0}" = 0 ] && { printf '  %-22s not running -- skipped\n' "$name"; return 0; }
    checked=$((checked + 1))
    local want_argv have_argv want_env v val
    # What the unit gets, not what this shell would pass on: systemd gives a unit no KOTEKAN_SRC,
    # and the broker's definition sets its own.
    want_argv=$(env -u KOTEKAN_SRC GNSS_CHAIN="$chain" "$here/run_component.sh" --print "$name" | awk -F'\t' '$1=="argv"{print $2}')
    have_argv=$(tr '\0' '\n' < "/proc/$pid/cmdline")
    if [ "$want_argv" != "$have_argv" ]; then
        printf '  %-22s ARGV MISMATCH\n' "$name"
        diff <(echo "$want_argv") <(echo "$have_argv") | sed 's/^/      /'
        bad=$((bad + 1)); return 0
    fi
    want_env=$(env -u KOTEKAN_SRC GNSS_CHAIN="$chain" "$here/run_component.sh" --print "$name" | awk -F'\t' '$1=="env"{print $2}')
    local miss=""
    while IFS= read -r kv; do
        [ -z "$kv" ] && continue
        # -x: the whole NUL-separated record. A substring match would let VAR=16 satisfy VAR=1,
        # and XVAR=1 satisfy VAR=1.
        grep -qzxF "$kv" "/proc/$pid/environ" 2>/dev/null || miss="$miss $kv"
    done <<< "$want_env"
    if [ -n "$miss" ]; then
        printf '  %-22s ENV MISSING:%s\n' "$name" "$miss"
        bad=$((bad + 1)); return 0
    fi
    printf '  %-22s ok (argv %d args, env %d vars)\n' "$name" \
        "$(echo "$want_argv" | grep -c .)" "$(echo "$want_env" | grep -c .)"
}

if [ $# -gt 0 ]; then
    case "$1" in
    obs) for c in $("$here/stack_components.sh" >/dev/null 2>&1; . "$here/stack_components.sh"; gnss_chains); do
             check obs "gnss-obs@$c.service" "$c"; done ;;
    *)   check "$1" "gnss-$1.service" ;;
    esac
else
    check broker     gnss-broker.service
    check gather     gnss-gather.service
    check aggregator gnss-aggregator.service
    check viewer     gnss-viewer.service
    check rail       gnss-rail.service
    . "$here/stack_components.sh"
    for c in $(gnss_chains); do check obs "gnss-obs@$c.service" "$c"; done
fi
echo "  checked $checked component(s), $bad mismatch(es)"
if [ $# -eq 0 ] && [ "$checked" -eq 0 ]; then
    echo "  nothing checked: no component of the stack is running here"
    exit 1
fi
[ "$bad" -eq 0 ]
