# systemd units for the GNSS infrastructure on `gnss.site.chord-observatory.ca`

**Installed on gnss as systemd user units** by `install_user_units.sh`, which copies the files
here into `~/.config/systemd/user` and reloads. The copies differ only in the checkout they
name: the installer's own in place of `/home/kvand/gnss/kotekan`. Edit them here, never the
installed copies.

**They carry `ConditionHost=` (gnss only), and must.** The user copies live in the NFS-shared
home, so cf06, all eight cx nodes, recv1 and choco see them too. Without the condition, any login
on any of those hosts starts the whole stack there, crash-looping until the session closes; on
cf06, a backgrounded cube archiver holds its session open, so the strays loop for as long as the
archiver runs. Keep the condition on every unit, not only the target: a target's failed
condition does not cancel the start jobs its `Wants=` queued. To run the stack on another host,
change the condition there deliberately.

The settings are kept here so they are reviewable: each non-obvious one answers a fault rather
than a preference, and the comments say which.

Install or update (no root; reloads, restarts nothing):

```sh
sh install_user_units.sh
systemctl --user restart gnss-<unit>.service   # to pick up a changed unit
```

The service logs go to `/var/tmp/gnss-logs/` and are rotated by `../logrotate.conf` from the gnss
crontab.

`gnss-obs@.service` is templated on the chain name; `gnss-stack.target` pulls in the ten
instances the chain manifest lists. Keep that list and
`config/gnss_chains_chord.yaml` in step — the manifest is the armed authority, this is a mirror.

## The cf06 cube leg: `gnss-cube.target`

`gnss-cubearch.service` (the beam-cube archiver, `cubearch_up.sh --check` as its pre-flight) and
`gnss-cubecompact.service` (the compactor loop, which finishes its in-flight pass on a stop)
carry `ConditionHost=cf06` and hang off `gnss-cube.target`, not the stack target. The install
script copies them with the rest. The target is enabled, and the condition keeps it to cf06.
Enabling it stops nothing started by hand, so stop the by-hand archiver and compactor first:
two compactors on one L0 tree corrupt it.

```sh
systemctl --user daemon-reload && systemctl --user enable --now gnss-cube.target
```

Their logs stay where the launchers put them, `/tmp/gnss_cubearch.log` and
`/tmp/gnss_cubecompact.log` on cf06.

## Beside them, not of them

`claude-screen.service` in the same installed directory is the operator console (the Claude Code
sessions in the `claude` screen, resumed with Remote Control at boot). It is installed by hand
from `~/.config/claude-screen`, not by `install_user_units.sh`, and is not part of the stack.
