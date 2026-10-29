#!/bin/bash
# Run one GNSS component in the foreground, from the shared definition. This is what systemd
# execs.
#
# Foreground and `exec` are the point: the process systemd supervises must be the component
# itself, so Restart=, MemoryMax= and a clean stop all act on it. The header of
# stack_components.sh says why the units cannot call a daemonising launcher instead.
#
#     run_component.sh broker|gather|aggregator|viewer|rail|lanerelay
#     GNSS_CHAIN=gps_l5 run_component.sh obs
#     run_component.sh cubearch|cubecompact   # cf06's cube leg (gnss-cube.target)
#     run_component.sh --print <name>     # show argv + env, run nothing (what the gate uses)
set -eu

here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=stack_components.sh
. "$here/stack_components.sh"

PRINT=0
[ "${1:-}" = "--print" ] && { PRINT=1; shift; }
name=${1:?usage: run_component.sh [--print] <broker|gather|aggregator|viewer|obs|rail|lanerelay|cubearch|cubecompact>}

gnss_component "$name"
[ -n "${GNSS_CWD:-}" ] && cd "$GNSS_CWD"

if [ "$PRINT" = 1 ]; then
    printf 'cwd\t%s\n' "${GNSS_CWD:-$PWD}"
    for a in "${GNSS_EXEC[@]}"; do printf 'argv\t%s\n' "$a"; done
    # Every variable the definition may export. A name missing from this list is a variable
    # stack_contract_gate.sh cannot check, which is how an environment drifts unnoticed. Read
    # as shell variables, not from printenv: one the definition sets but forgets to export is
    # still listed, so the gate finds it missing from the process (#134).
    for v in GNSS_PY GNSS_CHAIN KOTEKAN_SRC \
             OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS; do
        eval "val=\${$v:-}"; [ -n "$val" ] && printf 'env\t%s=%s\n' "$v" "$val"
    done
    exit 0
fi

# The gather's and the aggregator's interface preflight (iface_preflight.sh): before the exec,
# so the process systemd supervises keeps its argv and environment.
if [ -n "$GNSS_IFACE_ROLE" ] && ! gnss_iface_preflight "$GNSS_IFACE_ROLE" "${GNSS_EXEC[2]}" \
        timeout 20 "${GNSS_EXEC[0]}" --interfaces-json; then
    echo "run_component: $name refused by the interface preflight (above;" \
         "GNSS_SKIP_IFACE=1 overrides)" >&2
    exit 1
fi

shift                       # anything after the component name is appended verbatim
exec "${GNSS_EXEC[@]}" "$@"
