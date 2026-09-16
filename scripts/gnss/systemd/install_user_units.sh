#!/bin/sh
# Install the stack as systemd USER units on this host. No root anywhere.
#
# WHY A DERIVATION AND NOT A SECOND COPY OF THE UNITS: the .service files beside this script are
# the canonical form and the place the hard-won settings are explained. Two hand-maintained
# copies would agree on the day they were written and diverge on the first edit, and the
# settings that matter here are exactly the ones nobody re-derives (GNSS_PY, LimitNOFILE, the
# gather's long RestartSec). So the user variant is generated, every time, from those.
#
# Three differences, and only three:
#   - no User=            a user unit already runs as the user
#   - default.target      user units have no multi-user.target
#   - logs to a LOCAL dir /var/tmp/gnss-logs, not /var/log (no root) and NOT the NFS home
#     (259 MB/h of appends over NFS, and O_APPEND is not atomic there)
#
# Converting to system units once passwordless sudo exists is: install the originals into
# /etc/systemd/system, create /var/log/gnss, drop in gnss.logrotate. Nothing else changes.
set -eu
SRC=$(cd "$(dirname "$0")" && pwd)
DST=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
LOGS=${GNSS_LOG_DIR:-/var/tmp/gnss-logs}

mkdir -p "$DST" "$LOGS"
for f in "$SRC"/gnss-*.service "$SRC"/gnss-*.target; do
    b=$(basename "$f")
    sed -e '/^User=/d' \
        -e "s|WantedBy=multi-user.target|WantedBy=default.target|" \
        -e "s|append:/var/log/gnss/|append:$LOGS/|" \
        "$f" > "$DST/$b"
done
systemctl --user daemon-reload
echo "installed into $DST:"
ls -1 "$DST" | sed 's/^/  /'
echo "logs -> $LOGS"
echo
echo "Linger must be on for these to start at boot and survive logout:"
loginctl show-user "$(id -un)" -p Linger
