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
   (chain, PRN), apply the `fleet-lane-transfer` table (`CHAIN:LO-HI` MHz, the chain by its
   broker key, the same form as `lane-transfer`) to choose the target channels, and
   `POST .../set_lanes` to each GPU the lanes whose targets meet its live channels, less the
   lanes it measured itself. The pooled per-element instrumental delays ride along every ten
   seconds. The relay reads the fleet
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

`fleet-lane-transfer-bright` (same form) holds the targets for a close pass: a lane whose
power is `fleet-lane-transfer-bright-db` (default 15) above the median of its chain's lanes in
the same cycle also goes there, on every GPU including the one that measured it. Those targets
are the sidelobes of wideband signals (B3I and L3OC at 10.23 Mcps, GLONASS L2's P code) that a
satellite near the boresight spreads far past its own lane's channels; without them only the
eigen path removes it there. No table, no change.

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
   `fleet-lane-transfer: [gps_l2c:1217.4-1237.8, gal_e6:1263.4-1273.6]` (the relay's table, the
   same form as `lane-transfer`; the targets are already in the projected ranges). Regenerate,
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
gnss-lanerelay` (in gnss-stack.target; `run_component.sh lanerelay`). Stop the canary's relay
first (`systemctl --user stop gnss-lanerelay-canary`, the transient unit run on 2026-10-30): the
target wants gnss-lanerelay, so a daemon-reload and a target start bring the fleet relay up beside
it, and two relays post to the same GPUs. Then
the transfers beside the lobes (2026-10-30, fleet-94daacb22): the projected range is 5939-6626
(1160.0-1294.1 MHz, one with the SK exemption), with the eigen path kept on the lobes
(`n2-project-eigen-channels`; the stage runs on one core per GPU, and the lane projection's
n^2 k per channel-frame already puts it at 77-86 % with 43 channels and 20-30 lanes). The
local table carries `l2of` over the GLONASS L2 P-code spill, `e6` over the PRS upper lobe and
`e5a`/`b2a`/`l5` over the lower AltBOC edge. The relay's table stays with the lanes that are
not on every GPU (L2C, each GLONASS L2OF satellite) and E6: the relay's copies of a GPU's own
satellites pass the duplicate test on a far target (the re-steer over 10-16 MHz differs by the
pooled delays' noise) and filled the 40-lane budget on the E5/B2 edges with near-duplicates.

## The instrumental phase table (#246)

**Why.** A lane carried from the channel it was measured on (f_src) to another (f) needs each
input's instrumental phase change between the two as well as the geometric one. Until now that
part depended on the GPU: a chain's own transfers used its own `InstDelay` (which needs two or
more of its channels on the GPU, so gps_l2c had none and carried geometry only), and the relay's
copies used the pooled gps_l5 delays (12.6 ns rms off at G03's 11-25 pass). Measured from the Tau
A calibration it is one number per input and channel: one delay per input across 1160-1295 MHz
leaves 6-10 deg rms (2026-11-26, cal_1126). On the 11-25 G03 lane dump a gps_l2c lane carried
12 MHz reads overlap^2 0.96 with the calibration's phase against 0.91-0.92 geometry only and
0.93-0.94 with one delay per input (`fixtures/deproj_1125/xfer_phase_test.py`); 1.35 ns rms per
input is left, cause not known.

**The table** (`python/scripts/gnss/inst_phase_table.py`): the calibration's phase per input at
knots (its solved channels, at most one per 0.8 MHz, 1150-1300 MHz), with the geometry toward
Tau A taken off, in the frame of the science N^2's eigenvectors (the lane's frame after
`lane_conj`), and the F-engine start it was made under (`made_frame0`). Built after a transit:
`inst_phase_table.py build CAL.npz fixtures/obs/inst_phase/current.json`. The cal_1125b table
turned by the 11-25 start's step row matches the cal_1126 one to 2.8 deg rms (99.5 unturned).

**Relay.** With `--inst-phase FILE` the relay turns the table to the newest start in the frame0
file by the step table (`fengine_steps.steps_between`, with half_step) and adds it to `set_lanes`
when it changes and every 300 s. While a start in between is unmeasured it sends an empty one.

**Contract** (`set_lanes`, optional):

    "inst_phase": {"made_frame0": <frame0_int>, "fids": [<knot freq_id>, ...] (ascending),
                   "phase_deg": [[<n> entries, null where the knot does not hold the input], ...]}

Absent leaves the node's table as it is; `{}` clears it. The whole body is refused (400) for a
row of another length than the tracker elements, fids that are not ascending integers, a row
count other than the fids', no knots, a made_frame0 that is not an integer, or a phase that is
neither a number nor null.

**Node** (GnssN2Project, `lane_inst_phase: true`; needs `fleet_lanes`). Per element, the phase
at a channel is linear in freq_id between the two nearest knots that hold the element (the
nearest knot's value outside them; `gnss::InstPhaseTable` in gnssFleetLane.hpp, the same as
`inst_phase_table.phase_at`); an element no knot holds has none. A lane carried from f_src to f
is multiplied, per station in the N^2 frame, by exp(i(phi(f) - phi(f_src))) where the element has
both, and by the pooled delay where it does not (as before). This applies to the relay's copies
(in place of the pooled delay in `fleet_lane_direction`) and to the board's own transfers: a
transferred `ProjEntry` records the channel it was measured on (`fid_src`), and the stage applies
the same factor to it. So every GPU carries a satellite's lane the same way, whichever GPU
measured it.
- The assemblers then must not apply their own delays to their transfers: `lane_transfer_inst_apply:
  false` (the generator's `--lane-inst-phase`, the fleet yaml's `lane-inst-phase: true`, sets it
  for every assembler and `lane_inst_phase` on the stage, so one regeneration switches both);
  their `InstDelay` keeps estimating, for `get_lane_inst` and the relay's pooled fallback.
- With no table (none posted, or cleared after a start), a transfer on the board gets the pooled
  delay, as a relay copy does, so turning the assemblers' own delays off loses nothing.
- Whether the table holds an element does not depend on the channel, so each element takes the
  table, else the pooled delay, else nothing, at both ends. The stage keeps a phasor per element,
  u(f) (`gnss::inst_carry_phasors`), for each live channel and each relay source channel,
  rebuilt when the table or the delays change; the carry is u(f) conj(u(f_src)). The pooled
  delay is a raw-frame slope (InstDelay::apply turns q by exp(+i 2 pi df tau)), so with
  `lane_conj` it turns the N^2-frame lane by exp(-i 2 pi df tau).
- `lane_inst_phase: false` (the default): a posted table is kept and shows in the gauges, and
  nothing else changes.
- The table ages out: when the last post that carried it is older than
  `lane_inst_phase_max_age_s` (900 s, three of the relay's reposts; 0 = never) the stage clears
  it as `{}` does and logs a WARN, so a dead relay cannot leave a table from before an F-engine
  start. The pooled `inst_delay_ns` does not age out (unchanged): with the relay dead the board's
  transfers keep the last delays.
- Gauges: `kotekan_gnss_n2project_inst_phase_knots` (0 = no table), `..._inst_phase_age_s` (since
  the table was last posted; -1 with none).

**Check.** A lane dump at a bright pass: a carried lane's overlap with the top mode against
distance from its source (G03 table, `fixtures/deproj_1125/G03_LANE_DUMP.md` section 6); and
`sat_fit.py` on the GPS L2C satellites at 1216.8 and 1238.7 MHz (9-12% left on 11-26 05-07Z).
