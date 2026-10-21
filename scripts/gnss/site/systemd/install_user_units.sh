#!/bin/sh
# Install the stack as systemd user units. No root anywhere.
#
#     sh install_user_units.sh          install into ~/.config/systemd/user and reload
#     sh install_user_units.sh <dir>    only write the units into <dir> (no log dir, no reload),
#                                       to diff them against what is installed
#
# The .service and .target files beside this script are the units, installed as they are but
# for one thing: /home/kvand/gnss/kotekan, the checkout they name, becomes the checkout this
# script is in, so a checkout elsewhere installs units that run its own scripts. They are the
# canonical form and where the settings are explained, so the installed copies are never
# edited by hand. They run as the user (no User=) and hang off default.target. The stack units
# log to a local dir, /var/tmp/gnss-logs (cf06's cube units to /tmp): not /var/log (no root)
# and not the NFS home (heavy appends, and O_APPEND is not atomic over NFS). The install dir is
# in the NFS-shared home, so every host sees the units; each unit's ConditionHost= keeps it to
# its own host.
set -eu
SRC=$(cd "$(dirname "$0")" && pwd -P)
K=$(cd "$SRC/../../../.." && pwd -P)
DST=${1:-${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user}
LOGS=/var/tmp/gnss-logs

k=$(printf '%s\n' "$K" | sed 's/[\\&|]/\\&/g')    # a sed replacement
mkdir -p "$DST"
for f in "$SRC"/gnss-*.service "$SRC"/gnss-*.target; do
    sed "s|/home/kvand/gnss/kotekan/|$k/|g" "$f" > "$DST/$(basename "$f")"
done
if [ $# -gt 0 ]; then
    exit 0
fi
mkdir -p "$LOGS"
systemctl --user daemon-reload
echo "installed into $DST:"
ls -1 "$DST" | sed 's/^/  /'
echo "logs -> $LOGS"
echo
echo "Linger must be on for these to start at boot and survive logout:"
loginctl show-user "$(id -un)" -p Linger
