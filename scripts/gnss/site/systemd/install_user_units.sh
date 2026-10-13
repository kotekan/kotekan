#!/bin/sh
# Install the stack as systemd USER units on this host. No root anywhere.
#
# The .service and .target files beside this script ARE the units, and are installed verbatim:
# they are the canonical form and the place the hard-won settings are explained, so the
# installed copies are never edited by hand (two hand-maintained copies agree on the day they
# are written and diverge on the first edit). They run as the user (no User=), hang off
# default.target, and log to a LOCAL dir, /var/tmp/gnss-logs -- not /var/log (no root) and NOT
# the NFS home (259 MB/h of appends over NFS, and O_APPEND is not atomic there).
set -eu
SRC=$(cd "$(dirname "$0")" && pwd)
DST=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
LOGS=/var/tmp/gnss-logs

mkdir -p "$DST" "$LOGS"
for f in "$SRC"/gnss-*.service "$SRC"/gnss-*.target; do
    cat "$f" > "$DST/$(basename "$f")"
done
systemctl --user daemon-reload
echo "installed into $DST:"
ls -1 "$DST" | sed 's/^/  /'
echo "logs -> $LOGS"
echo
echo "Linger must be on for these to start at boot and survive logout:"
loginctl show-user "$(id -un)" -p Linger
