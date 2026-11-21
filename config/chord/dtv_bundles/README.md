# DTV detector bundles

Each directory is a PilotProxy runtime bundle, named `<purpose>-<first 8 digits of the sha256 of sha256sums.txt>`. Set `dtv_runtime_bundle_dir` in `dtv.j2` to the installed path of one directory. A bundle is never edited in place, so a new calibration gets a new directory. To make one, see `docs/RUNNING_THE_DETECTOR.md` in the [PilotProxy repository](https://github.com/WVURAIL/pilotproxy).

| Bundle | Channels | Use |
|---|---|---|
| `pathfinder-commissioning-b8026807` | 18, 24, 29 | Detect-only runs, never with `/dtv_mask_enabled` on |
| `pathfinder-commissioning-e2e90219` | 18, 24, 29, control 37 | The same words, and records channel 37 for the next calibration. Detect-only runs, never with `/dtv_mask_enabled` on |
