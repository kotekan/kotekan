#!/bin/sh
# Block until the broker's chain inventory has SETTLED, for use as a systemd ExecStartPre.
#
# WHY: livebeam_server.py discovers the chain list ONCE at construction, with a 3 s timeout, and
# silently falls back to a static single-band table when the fetch is short or fails. systemd's
# After= orders START, not READINESS, so under systemd the viewer raced the broker's warm-up and
# cached whatever existed at that instant -- it came up showing 2 chains once and 1 chain the
# next time, against the 8 the fleet runs. Started by hand, minutes after the broker, it had
# never raced.
#
# "Settled" is deliberately not "8": the chain count is a property of the manifest, not of this
# script, and hard-coding it here would make a legitimate chain change look like a failure. A
# count that stops changing is the honest signal.
set -u
URL=${GNSS_BROKER_URL:-http://127.0.0.1:12060}/get_chains
STABLE_FOR=${GNSS_CHAINS_STABLE_S:-15}      # count must hold this long
DEADLINE=${GNSS_CHAINS_TIMEOUT_S:-300}

count() { curl -s -m5 "$URL" 2>/dev/null | tr ',' '\n' | grep -c '"chain"' || echo 0; }

start=$(date +%s); last=-1; since=0
while :; do
    now=$(date +%s)
    [ $((now - start)) -ge "$DEADLINE" ] && {
        echo "wait_for_broker_chains: giving up after ${DEADLINE}s at $last chain(s) -- starting anyway" >&2
        exit 0        # ⚠️ EXIT 0 ON PURPOSE: a degraded viewer beats no viewer, and the unit
    }                 #    must not crash-loop because the broker is slow.
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
