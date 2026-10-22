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
#
# Installing from a checkout other than the one the installed units run moves the whole stack
# to it at its next restart, so that is refused unless the checkout is named:
#     GNSS_UNITS_CHECKOUT=<this checkout> sh install_user_units.sh
set -eu
if [ $# -gt 0 ] && [ -z "$1" ]; then
    echo "install_user_units.sh: empty <dir>" >&2
    exit 2
fi
SRC=$(cd "$(dirname "$0")" && pwd -P)
K=$(cd "$SRC/../../../.." && pwd -P)
DST=${1:-${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user}
LOGS=${GNSS_LOGS:-/var/tmp/gnss-logs}   # the stack's log dir (the tests point it elsewhere)

if [ $# -eq 0 ]; then
    run=$(sed -n 's|^ExecStart=\(.*\)/scripts/gnss/site/run_component.sh broker$|\1|p' \
        "$DST/gnss-broker.service" 2>/dev/null || true)
    if [ -n "$run" ] && [ "$run" != "$K" ] && [ "${GNSS_UNITS_CHECKOUT:-}" != "$K" ]; then
        echo "REFUSING: the installed units run $run, and this checkout is $K." >&2
        echo "To move the stack here: GNSS_UNITS_CHECKOUT=$K sh $0" >&2
        exit 1
    fi
fi

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
echo "installed into $DST, running $K:"
ls -1 "$DST" | sed 's/^/  /'
echo "logs -> $LOGS"
echo
echo "Linger must be on for these to start at boot and survive logout:"
loginctl show-user "$(id -un)" -p Linger
