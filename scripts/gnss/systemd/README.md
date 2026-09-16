# systemd units for the GNSS infrastructure on `gnss.site.chord-observatory.ca`

⚠️ **NOT INSTALLED, AND NOT INSTALLABLE YET.** They need two things first: the VM resized to
4-6 vCPU with a CPU model that is not QEMU's 2.5+ baseline, and a kotekan built
`-DARCH=x86-64-v2 -DUSE_CUDA=OFF`. Today's cf06 binary dies on `SIGILL` there. See
[`docs/CHORD_GNSS_VM_MIGRATION.md`](../../../docs/CHORD_GNSS_VM_MIGRATION.md).

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

`gnss-obs@.service` is templated on the chain name; `gnss-stack.target` pulls in the eight
instances the chain manifest lists. Keep that list and
`config/gnss_chains_chord.yaml` in step — the manifest is the armed authority, this is a mirror.
