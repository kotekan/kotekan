# DTV detector bundles

Each directory is a PilotProxy runtime bundle, named `<purpose>-<first 8 digits of the sha256 of sha256sums.txt>`. Set `dtv_runtime_bundle_dir` to the absolute path of one directory. A bundle is never edited in place, so a new calibration gets a new directory. To make one, see `docs/RUNNING_THE_DETECTOR.md` in the [PilotProxy repository](https://github.com/WVURAIL/pilotproxy).

| Bundle | Channels | Use |
|---|---|---|
| `pathfinder-commissioning-5547008e` | 18 (weak), 24 (strong), 29 (quiet) | Detect-only runs, never with `dtv_apply_mask` |
