# scripts/gnss/site

Deployment and operations tooling for the CHORD GNSS installation: the launchers, the systemd
user units, the cron jobs, on-sky probes and gates, and local development helpers. These scripts
encode this site's hosts, paths and fleet. They are not part of the upstream GNSS pull requests;
the tools one level up are.

The launchers and cron jobs find the checkout they sit in from their own path, and the kotekan
tree (its builds and headers) through `kotekan_src.sh`, by the rule
`python/scripts/gnss/kotekan_src.py` follows; `KOTEKAN_SRC` overrides it.

Things outside the repository that name these paths, and must follow a move:

* the installed user units on gnss: re-run `systemd/install_user_units.sh`;
* the gnss crontab: `eop_cron.sh`, `bad_inputs_cron.py`, `elem_ref_cron.sh`, `tle_refresh.py`;
* the cf06 crontab: `chain_health_cron.sh`;
* the cf06 compactor loop, which finds `beamcube_daily.sh` beside itself: `gnss-cube.target`
  runs it on cf06, so restart it there with `systemctl --user restart gnss-cubecompact`.
