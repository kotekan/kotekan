# systemd units for the GNSS infrastructure on `gnss.site.chord-observatory.ca`

**Installed on gnss as systemd USER units since 2026-09-17**, by `install_user_units.sh`, which
copies the files here into `~/.config/systemd/user` verbatim and reloads. Edit them here, never
the installed copies.

⚠️ **They carry `ConditionHost=` (gnss only), and must.** The user copies live in the NFS-shared
home, so cf06, all eight cx nodes, recv1 and choco see them too. Before 2026-10-02 any login on
any of those hosts started the whole stack there, crash-looping until the session closed. On
cf06, a backgrounded cube archiver held its session open, so the strays kept looping for as
long as the archiver ran. Keep the condition on every unit, not only the target: a target's
failed condition does not cancel the start jobs its `Wants=` queued. To run the stack on another
host, change the condition there deliberately.

They are in the repo now so the settings are reviewable, because each non-obvious one is a
fault we have already paid for rather than a preference — the comments say which.

Install or update (no root; reloads, restarts nothing):

```sh
sh install_user_units.sh
systemctl --user restart gnss-<unit>.service   # to pick up a changed unit
```

The service logs go to `/var/tmp/gnss-logs/` and are rotated by `../logrotate.conf` from the gnss
crontab.

`gnss-obs@.service` is templated on the chain name; `gnss-stack.target` pulls in the eight
instances the chain manifest lists. Keep that list and
`config/gnss_chains_chord.yaml` in step — the manifest is the armed authority, this is a mirror.
