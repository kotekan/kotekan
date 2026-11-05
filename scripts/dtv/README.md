# DTV subtraction tools

Operations and analysis for `cudaDtvSubtract` (blind beam template + per-input lagged fit, rolling g).
Run on cf05/cf06 with `nice`, never on the gnss VM; REST-only tools (`dtv_roll.py status|arm|...`) are safe anywhere.
Several tools still use the paths of their first home, `~/gnss/bb1030` and `~/gnss/dtv-wt/scripts/gnss`
(`dtv_state.py`, `beam_dump_read.py`).

**Live stage**
- `dtv_live.py`: one node: `setup` (eigen start, window, fit, load per source), `refit`, `load`, `window`, guards.
- `dtv_fleet2.sh`: `setup [nodes]` in parallel over the fleet (states in `builds/dtv-live/states-rev2`).
- `dtv_roll.py`: rolling-g REST control: `arm`, `disarm`, `freeze`, `status`, `save`, and `watch` (saves, restores
  restarted nodes, Tau A freeze). Saves go through the node's `/tmp/gnss/dtvroll` (kotekan is root, /home is NFS).
- `dtv_canary.py`: residual/raw per channel of both GPUs of one node from short windows.
- `dtv_roll_night.py`: the watch log as a night of stage telemetry.

**Judging (recv1 N2 subset)**
- `vis_dtv_extract.py` (eigenvalues, Tau A template excess per 10 s bin), `vis_dtv_plot.py` (Tau A transit),
  `vis_dtv_nights.py` (nights side by side), `vis_dtv_modes.py`, `vis_dtv_sawtooth.py`, `vis_dtv_gdrift.py`,
  `vis_dtv_gshort.py`.

**Surveys and the 10-30 baseband dump**
- `dtv_gain_survey.py` (transmitters from the F-engine gains), `rfi_modes.py` (spatial modes per channel),
  `bb_extract.py` (pull channels from the dump), `dtv_cov.py`, `dtv_modes.py`, `dtv_blind*.py`, `dtv_protect.py`.
- `atsc_*.py`: 8-VSB acquisition, equalisation, trellis decode, replica regression; `atsc_nsrc.py` (sources needed).

**Replay**: `dtv_replay.cu`, `dtv_replay*.sh`, `dtv_replay.py`, `dtv_synth_state.py` (GPU cost on cf05).
