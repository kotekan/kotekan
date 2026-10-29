# Fleet lanes: lane directions shared across nodes (#190 tier 1.5, fleet-wide)

## Why

A lane is a tracked satellite's per-element response on one channel (gnssLaneDirection.hpp).
GnssN2Project nulls it in the N^2 frame, and `lane_transfer_freq_ids` carries it to the same
satellite's other signals (GPS L2 P(Y)/M from L2C, the Galileo E6 PRS lobe from E6-B/C).
Both happen inside one GPU's process, through the ProjBoard, because the lane is measured
from the assembler's prompt rows and only that GPU's chains write the board. So a transfer
reaches only the GPUs that track the satellite in the source band: GPS L2C occupies six
channels, one GPU per node holds one, and only five of the fourteen projecting stream phases
have GPS lanes in the L2 band (docs/CHORD_TAUA_GUOWANG_20261029.md section 2).

The gather and broker fit across the whole spectrum, but they see beamformed records: no
per-element vector leaves a node today.

## What

A lane is shipped between nodes with the track it was steered with, and re-steered at the
receiver.

The assembler publishes `q = conj(D(f_src, e)) m / |m|`: `m` is the steered prompt-row mean
(the per-element instrument response, slowly varying), `D(f, e)` the geometric phasor of
ElemSteer::build for line of sight `e` at frequency `f`. The local transfer to `f` applies
`conj(shift(f - f_src))` and the instrumental delays, which is `conj(D(f, e)) D(f_src, e) q`
times `exp(i 2 pi (f - f_src) tau_e)`. The same expression evaluates on any node that knows
the element positions, the steering sign, `e(t)` and `tau_e`; all four are fleet-wide
quantities, and `e` moves along a track the broker already posts.

Three pieces:

1. **Node, source side.** Every lane entry on the ProjBoard carries the line of sight and its
   rate at the instant the steering table described (`ElemSteer::track`), and whether it is a
   transfer (`xfer`). `GET gnss_n2_project/proj_<g>/get_lanes` returns, per owner, the measured
   entry of best coherence on this GPU's board. The gps_l5 assembler (on every GPU, with
   transfers) serves its InstDelay estimate at `GET .../get_lane_inst`.

2. **Relay** (`scripts/gnss/site/lane_relay.py`, `gnss-lanerelay.service` on the gnss VM).
   Once a second: `get_lanes` from every projecting GPU, keep the best copy of each
   (band tag, PRN), apply the `fleet-lane-transfer` table (`TAG:LO-HI` MHz, the same form as
   `lane-transfer`) to choose the target channels, and `POST .../set_lanes` to each GPU the
   lanes whose targets meet its live channels, less the lanes it measured itself. The pooled
   per-element instrumental delays ride along every ten seconds. The relay reads the fleet
   configuration and the generated node configs and nothing else; it never touches the
   control path. If it dies, fleet lanes age out of every node in `fleet_lane_max_age_s` and
   projection continues with the local lanes: today's state.

3. **Node, receiving side.** `GnssN2Project` keeps the posted set (`gnssFleetLane.hpp`). Every
   `fleet_lane_rebuild_s` it evaluates each lane's direction on each target channel it writes,
   at the frame's UTC with the line of sight extrapolated along its track, and maps it onto
   the stations axis. `build_lanes` appends those directions after the board's, in the
   posted order, so a local lane always wins the Gram-Schmidt duplicate test and the
   `k_lanes_max` budget, and a fleet lane for a satellite this GPU tracks itself is dropped as
   a duplicate.

## Contract

`set_lanes` body (one object):

    {"t_utc": <unix s>,
     "inst_delay_ns": [<n>] (optional; the pooled per-element instrumental delay),
     "lanes": [{"id": "<node>/<owner>", "f_src_hz": <Hz>, "t_utc": <unix s>,
                "los": [E, N, U], "los_rate": [E, N, U] (per s), "coh": c, "power": p,
                "fids": [[lo, hi], ...], "q": [[re, im], ...] (n entries, unit norm)}, ...]}

`n` is the tracker element count (`tracker_stations`); a body of another length is refused
(400). A POST replaces the whole set; `inst_delay_ns` absent leaves the delays as they are,
empty clears them. Targets outside this GPU's live channels are ignored.

`get_lanes` reply: `{"ready", "n", "lanes": [{"owner", "freq_id", "t_utc", "los",
"los_rate", "coh", "power", "q"}]}` (`ready` false, and no lanes, until the first frame has
labelled the channels); `get_lane_inst`: `{"warm", "weight", "rms_ns", "df_hz",
"delay_ns": [<n>]}`.

Gauges: `kotekan_gnss_n2project_fleet_lanes` (held), `..._fleet_lane_age_s` (oldest held),
`..._fleet_lane_k{freq_id}` (fleet lanes in the basis this frame).

## Config

Generator `--fleet-lanes` (fleet yaml `fleet-lanes: true`): GnssN2Project gets `fleet_lanes`,
`elem_positions_enu` and `elem_steer_sign` (the assemblers' values). The relay's transfer
table is `fleet-lane-transfer` in the fleet yaml's common block, read by the relay only.
The target channels must be in `n2-project-live-freq-ids`, as for the local transfers.

## Rollout

Done 2026-10-29 (branch kv/fleet-lanes, two commits on 0ca8ce6e8; binary
`~/gnss/builds/fleet-a87913c14/kotekan/kotekan`, Release, libraries resolve from a node):
boost `test_gnss_lane_direction` (the fleet form equals the local transfer to 2e-8 with the
delays; 20 s along the track the re-steered direction is exact where a held one is off by
0.33); the generator test with `--fleet-lanes`; a cx51 preview config, `--check-config`
PASSED; `config/ci-tests/cpu_batch/run_n2accum_gnssproject_fleet.yaml` run by hand on cf05
with `get_lanes` read and `set_lanes` posted during the run (lanes in `fleet_lane_k` behind
the board's, a short body refused with 400, the set aged out at 10 s). A `--dry-run` is not
possible on cf05 (DPDK EAL): the canary is the first real construction.

### Canary (one node; the fleet is back ~2026-10-31)

1. The python half must be in the checkout the stack regenerates from:
   `grep -c fleet-lanes ~/gnss/gnss-broker/config/gen_chord_gnss_config.py` (0 = not yet;
   the cleanup session extracts from the branch). The relay itself can run from the
   worktree with `--root ~/gnss/gnss-broker`.
2. Fleet yaml, `common:`: `fleet-lanes: true` and
   `fleet-lane-transfer: [l2c:1217.4-1237.8, e6:1263.4-1273.6]` (the relay's table, the same
   form as `lane-transfer`; the targets are already in the projected ranges). Regenerate,
   `--check`.
3. Pick a node whose one GPU holds an L2C channel and the other does not, so the cross-GPU
   case tests itself with one node: cx51 (GPU 1 holds L2C; GPU 0 receives). Cycle it onto the
   binary: `GNSS_BIN=/home/kvand/gnss/builds/fleet-a87913c14/kotekan/kotekan
   scripts/gnss/site/node_up.sh cx51 restart` from ~/gnss/gnss-broker. Both GPUs are
   sources and receivers; every other node stays on the old binary (404 on `get_lanes`, so
   `--from cx51` keeps the log quiet).
4. Relay by hand (cf06 or the VM; ~50 MB):
   `/usr/bin/python3 ~/gnss/fleetlane-wt/scripts/gnss/site/lane_relay.py show --root
   ~/gnss/gnss-broker --from cx51 --to cx51`, then the same with `run --status
   ~/gnss/fixtures/obs/lane_relay`. `show` lists what each GPU serves and would receive.
5. Judge within minutes, on cx51:
   - `curl -s cx51:12048/metrics | grep n2project_fleet`: `fleet_lanes` > 0 on both GPUs,
     `fleet_lane_age_s` ~1-2 s, `fleet_lane_k{freq_id}` > 0 on GPU 0's L2 channels
     (6233-6338, phase 7) whenever GPS satellites are tracked on L2C.
   - `auto_proj_db` on those channels against GPU 1's and against the day before; recv1's
     cross-coherence on cx51's phase-7 streams in 1217-1238 MHz during a GPS pass
     (taua1029/taua_flux.py style, docs/CHORD_TAUA_GUOWANG_20261029.md section 2).
   - The stage's CPU (`n2-project-cpus` 15/19): the rebuild is ~100 lanes x 32 channels x 128
     phasors every 0.25 s, small; confirm.
   - `lane_inst` on a single node is one instance: the pooled delays are that one GPU's own
     estimate, no outlier rejection until three or more GPUs are on the binary.
6. Rollback at any step: stop the relay (the nodes age the set out in 10 s and keep their
   local lanes); `fleet-lanes: false`, regenerate and cycle; or the previous binary.

### Fleet

Every node on the binary, `fleet-lanes: true` everywhere, `systemctl --user start
gnss-lanerelay` (in gnss-stack.target; `run_component.sh lanerelay`). Then
`fleet-lane-transfer` grows: `glo_l2of` across 1237.9-1258.2 MHz once that range is in
`n2-project-live-freq-ids` (split the projection list from the SK-exemption list first), the
E6 PRS upper lobe, the B2/E5 edges.
