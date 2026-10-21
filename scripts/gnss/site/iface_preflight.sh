# The kotekan interface preflight, for the scripts that start kotekan: node_up.sh,
# stack_components.sh (through run_component.sh) and cubearch_up.sh. Sourced after
# kotekan_src.sh, never executed:
#
#     gnss_iface_preflight <role> <config> <command...>
#
# <command> prints the binary's `kotekan --interfaces-json`: the binary itself, or an ssh to the
# node that runs it. `python3 -m gnss_broker.iface preflight` (python/scripts/gnss/gnss_broker/
# iface.py) checks that document against the broker's interface golden on what <role> exchanges
# with the broker, and against the frame sizes <config> was generated for (its gnss_iface
# stamp). Its lines go to stderr.
#
# Returns 1 only when the check refuses the start. A binary built before the document, an
# unstamped config, a command that fails, or a check that cannot run warns and returns 0.
# GNSS_SKIP_PREFLIGHT=1 skips it.

gnss_iface_preflight() {
    local role=${1:?usage: gnss_iface_preflight <role> <config> <command...>}
    local cfg=${2:-} ij out rc=0
    shift 2
    if [ "${GNSS_SKIP_PREFLIGHT:-0}" = 1 ]; then
        return 0
    fi
    ij=$("$@" 2>/dev/null) || rc=$?
    # A binary that predates --interfaces-json prints "Invalid option" and exits non-zero:
    # that output goes to the check, which reads it as LEGACY. No output at all is a failure.
    if [ -z "$ij" ] && [ "$rc" -ne 0 ]; then
        echo "iface preflight ($role): WARN '$*' failed (exit $rc); the binary's interface is not checked" >&2
        return 0
    fi
    rc=0
    out=$(printf '%s\n' "$ij" | PYTHONPATH="$GNSS_CHECKOUT/python/scripts/gnss" \
          timeout 30 python3 -m gnss_broker.iface preflight --role "$role" \
          ${cfg:+--config "$cfg"} 2>&1) || rc=$?
    if [ -n "$out" ]; then
        printf '%s\n' "$out" >&2
    fi
    if [ "$rc" -eq 1 ]; then
        return 1
    fi
    if [ "$rc" -ne 0 ]; then
        echo "iface preflight ($role): WARN the check could not run (exit $rc); not checked" >&2
    fi
    return 0
}
