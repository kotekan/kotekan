#!/bin/sh
# Block until the broker's chain inventory has settled, for use as a systemd ExecStartPre.
#
# livebeam_server.py discovers the chain list once at startup, with a 3 s timeout, and serves
# whatever it got (or no chains) for its lifetime. After= orders the start, not readiness, so
# without this wait the viewer races the broker's warm-up and caches a partial list.
#
# "Settled" is a nonzero count that holds for GNSS_CHAINS_STABLE_S, not a fixed number: the
# chain count belongs to the chains manifest, and hard-coding it here would make a legitimate
# chain change look like a failure.
set -u
URL=${GNSS_BROKER_URL:-http://127.0.0.1:12060}/get_chains
STABLE_FOR=${GNSS_CHAINS_STABLE_S:-15}      # count must hold this long
DEADLINE=${GNSS_CHAINS_TIMEOUT_S:-300}

# grep -c prints its count even when it is 0 (and exits 1 then), so it needs no fallback; an
# `|| echo 0` would add a second line and break the numeric tests below.
count() { curl -s -m5 "$URL" 2>/dev/null | tr ',' '\n' | grep -c '"chain"'; }

start=$(date +%s); last=-1; since=0
while :; do
    now=$(date +%s)
    [ $((now - start)) -ge "$DEADLINE" ] && {
        echo "wait_for_broker_chains: giving up after ${DEADLINE}s at $last chain(s) -- starting anyway" >&2
        exit 0        # Exit 0 on purpose: a degraded viewer beats no viewer, and the unit
    }                 # must not crash-loop because the broker is slow.
    n=$(count)
    if [ "$n" -gt 0 ] && [ "$n" -eq "$last" ]; then
        [ $((now - since)) -ge "$STABLE_FOR" ] && {
            echo "wait_for_broker_chains: $n chain(s), stable ${STABLE_FOR}s after $((now - start))s"
            exit 0
        }
    else
        last=$n; since=$now
    fi
    sleep 2
done
