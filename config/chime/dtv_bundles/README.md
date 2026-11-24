# DTV detector bundles

Each directory is a PilotProxy runtime bundle, named `<purpose>-<first 8 digits of the sha256 of sha256sums.txt>`. Set `dtv_runtime_bundle_dir` to the absolute path of one directory. A bundle is never edited in place, so a new calibration gets a new directory. To make one, see `docs/RUNNING_THE_DETECTOR.md` in the [PilotProxy repository](https://github.com/WVURAIL/pilotproxy).

| Bundle | Channels | Use |
|---|---|---|
| `chime-hello-world-d1ed7815` | 29, 35 | First detect-only run, never with `dtv_apply_mask` |
| `chime-first-56005bb7` | 14 to 36, control 37 | Parameters only, to record the data for the first calibration, never with `dtv_apply_mask` |
