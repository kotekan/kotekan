# systemd units for the GNSS infrastructure on `gnss.site.chord-observatory.ca`

⚠️ **NOT INSTALLED — but no longer blocked.** The VM was re-provisioned on cf02 with 6 `host`
cores and an L40S passthrough, so the cf06 `ARCH=native` binary runs there as-is and the GPU
takes a real CUDA context. Install them when you want the cutover, following the order of work
in [`docs/CHORD_GNSS_VM_MIGRATION.md`](../../../docs/CHORD_GNSS_VM_MIGRATION.md) §8 --
infrastructure first, aggregator only after a peak-under-load sample.

They are in the repo now so the settings are reviewable, because each non-obvious one is a
fault we have already paid for rather than a preference — the comments say which.

Install, once the prerequisites are met:

```sh
sudo cp gnss-*.service gnss-*.target /etc/systemd/system/
sudo mkdir -p /var/log/gnss && sudo chown kvand /var/log/gnss
sudo cp gnss.logrotate /etc/logrotate.d/gnss
sudo systemctl daemon-reload
sudo systemctl enable --now gnss-stack.target
systemctl status 'gnss-*'
```

`gnss-aggregator.service` is **not** in `gnss-stack.target` on purpose: it is the last step of
the migration, not the first. Add it to the target when it actually moves.

`gnss-obs@.service` is templated on the chain name; `gnss-stack.target` pulls in the eight
instances the chain manifest lists. Keep that list and
`config/gnss_chains_chord.yaml` in step — the manifest is the armed authority, this is a mirror.
