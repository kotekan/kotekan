#!/bin/bash
# Prove that each RUNNING component still matches stack_components.sh -- argv AND environment.
#
# WHY THE ENVIRONMENT IS HALF THE POINT. #134: the systemd unit reproduced the broker's command
# line faithfully and dropped `export OPENBLAS_NUM_THREADS=1`, which is not on the command line.
# Nothing failed; the broker just spent two thirds of its cycles in OpenBLAS's busy-wait until
# perf found it. Comparing argv alone would have passed that. So this compares both.
#
#     stack_contract_gate.sh            # every component that is running
#     stack_contract_gate.sh broker     # just one
# Exits non-zero on the first mismatch, so it is usable as a gate.
set -u
here=$(cd "$(dirname "$0")" && pwd)
bad=0 checked=0

check() {
    local name=$1 unit=$2 chain=${3:-}
    local pid; pid=$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || echo 0)
    [ "${pid:-0}" = 0 ] && { printf '  %-22s not running -- skipped\n' "$name"; return 0; }
    checked=$((checked + 1))
    local want_argv have_argv want_env v val
    want_argv=$(GNSS_CHAIN="$chain" "$here/run_component.sh" --print "$name" | awk -F'\t' '$1=="argv"{print $2}')
    have_argv=$(tr '\0' '\n' < "/proc/$pid/cmdline")
    if [ "$want_argv" != "$have_argv" ]; then
        printf '  %-22s ARGV MISMATCH\n' "$name"
        diff <(echo "$want_argv") <(echo "$have_argv") | sed 's/^/      /'
        bad=$((bad + 1)); return 0
    fi
    want_env=$(GNSS_CHAIN="$chain" "$here/run_component.sh" --print "$name" | awk -F'\t' '$1=="env"{print $2}')
    local miss=""
    while IFS= read -r kv; do
        [ -z "$kv" ] && continue
        grep -qzF "$kv" "/proc/$pid/environ" 2>/dev/null || miss="$miss $kv"
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
[ "$bad" -eq 0 ]
