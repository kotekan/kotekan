#!/bin/sh
# Install the stack as systemd user units. No root anywhere.
#
# The .service and .target files beside this script are the units, installed verbatim: they are
# the canonical form and where the settings are explained, so the installed copies are never
# edited by hand. They run as the user (no User=) and hang off default.target. The stack units
# log to a local dir, /var/tmp/gnss-logs (cf06's cube units to /tmp): not /var/log (no root)
# and not the NFS home (heavy appends, and O_APPEND is not atomic over NFS). The install dir is
# in the NFS-shared home, so every host sees the units; each unit's ConditionHost= keeps it to
# its own host.
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
