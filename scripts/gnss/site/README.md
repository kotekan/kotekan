# scripts/gnss/site

Deployment and operations tooling for the CHORD GNSS installation: the launchers, the systemd
user units, the cron jobs, on-sky probes and gates, and local development helpers. These scripts
encode this site's hosts, paths and fleet. They are not part of the upstream GNSS pull requests;
the tools one level up are.

Things outside the repository that name these paths, and must follow a move:

* the installed user units on gnss: re-run `systemd/install_user_units.sh`;
* the gnss crontab: `eop_cron.sh`, `bad_inputs_cron.py`, `elem_ref_cron.sh`;
* the cf06 crontab: `chain_health_cron.sh`;
* the cf06 compactor loop, which finds `beamcube_daily.sh` beside itself: restart it with
  `cubecompact_up.sh`.
