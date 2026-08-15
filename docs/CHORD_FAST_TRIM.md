# The fast code loop: moving control to C++, leaving policy in Python (task #51)

**Status 2026-08-15.** Plan agreed with KV. F1 in progress.

## 1. The problem, stated as arithmetic

The code discriminator's authority is a **step per update**, not a rate:

    tau  = -clamp(disc, +-1)/4 * (spacing/0.5)      =>  |tau| <= 0.25 chips ALWAYS
    step = dll_gain * tau                            =>  <= 0.0625 chips at gain 0.25

against a measured **0.121 chips/s** of code drift on CHORD. So the loop needs
`0.121 / 0.0625 = 1.94` **steps per second** merely to break even, and the policy cycle
delivers one step every ~12 s. That is the 23x deficit of #51: the loop can pull in but
cannot hold, the prompt tap sits off-peak, and every statistic computed from it -- prompt
power, C/N0 coherent and incoherent, the deep fold, every coherence number -- is computed
on noise. It is one fault with a dozen symptoms.

No gain and no clamp fixes this. `tau` is clamped **by construction**, so the only lever is
the update rate. (Cutting the gain to "compensate" for a faster rate hands the entire win
straight back: same gain, faster rate.)

## 2. Three rates, and which one has to be fast

| rate | today | ceiling on the CURRENT wire |
|---|---|---|
| measurement (records on the wire) | 95.4 /s | 95.4 /s -- already there |
| integrator steps | ~3.1 /s | 95.4 /s |
| actuation (posts to trackers) | ~3.1 /s | 23.84 /s (frame-synced) |

A telemetry frame is 4 records x 2048 hops x 5.12 us = 41.94 ms, so frames arrive at
**23.84 Hz** and records at **95.4 Hz**. You cannot *measure* at 50 Hz. You do not need to:
each frame carries four records, which are four legitimate sequential integrator steps, and
because the limit is a step per update they can be folded into **one** post without losing
any authority.

    steps/s   slew authority    margin vs 0.121 chips/s
      1.94     0.121 chips/s    break-even
      3.1      0.194            1.6x    <- armed on gps_l5 since 2026-08-15
     23.84     1.49             12x     <- one step per frame
     95.4      5.96             49x     <- four steps per frame, ONE post

**Target: 95.4 steps/s, 23.84 posts/s.** Loop latency is ~2-3 frames (85-125 ms) which at
0.121 chips/s is ~0.015 chips of lag -- irrelevant, and the reason the loop does not need to
close on the node. If actuation at 95 Hz is ever wanted it is
`--telem-records-per-frame 1`, not a redesign.

## 3. The split

The rule that has to survive is **not** "Python owns the loop". It is *one place with the
fleet-wide view owns the loop*. An instance sees 7 of ~105 channels and one reference
element, which is exactly why `cudaGnssChordTrack`'s in-tracker `code_trim` -- which already
exists, and already runs at frame rate -- is default OFF. The gather host is the only other
place with the whole fleet at frame rate, so it can take the authority without violating
anything that mattered about the original rule.

**Python keeps the slow plane (the ~12 s cycle) -- everything that is policy or physics:**
BRDC fetch/parse and the ephemeris model; sky, visibility and boresight; the clock solve and
`state_filter` (b_sat, rrate, f_carrier); PRN assignment; presence and quality gates; who is
*armed*; publishing, viewer, archive; the transcript/equivalence gate. None of it belongs in
C++ -- it is all seconds-to-minutes work, and it is where every judgement lives.

**C++ takes the fast plane -- exactly three things:**
1. comb -> per-(chain, PRN) fleet E/P/L across all instances and all channels;
2. discriminator + leaky integrator + clamp;
3. the actuation post.

Nothing else. It invents no gate and chooses no PRN.

## 4. The seam

Two small interfaces, both JSON over the gather instance's existing REST server.

**Policy down** (Python -> controller, once per cycle):

    POST /<name>/set_policy
    { "chains": { "gps_l5": {
        "armed":   [4, 9, 27],
        "base_cp": {"4": 1234.56, ...},   // UNTRIMMED model code phase, chips
        "gain": 0.25, "leak_per_s": 0.155, "clamp": 3.0, "spacing": 0.5,
        "code_length": 10230, "q_floor": 2.2, "max_age_frames": 3 } } }

`base_cp` is the model phase; the trim is a correction **to the model**, preserved across
republication (the tracker's own trim comment says exactly this, and it is the reason a
re-seed must not zero it).

**State up** (controller -> Python, polled once per cycle):

    GET /<name>/get_state
    { "gps_l5": { "4": {"trim":..., "disc":..., "q":..., "n_steps":..., "n_railed":...,
                        "n_frames":..., "last_win":...} } }

so the viewer and the archive keep a single origin.

## 5. Where the C++ lives

A new stage **`GnssFleetTrim`, inside the gather instance**, as a second consumer of
`telem_buf`. Not a separate process reading the served byte stream: it needs exactly the
buffer that instance already has, and a second parse of the same stream can silently
disagree with what the broker saw -- a bug class this tree has paid for repeatedly (#52,
#53, #33 are all "the address was inferred").

⚠️ **THE COST OF THAT CHOICE.** A stalled second consumer back-pressures `telem_buf`, and
`bufferRecv` then drops frames **for the broker too**. So the controller must have a hard
per-frame budget and **skip, never queue**: mark the window missed, count it, move on. The
counter is not optional -- a silently-skipping controller looks exactly like a working one.

**Collation** stays the exact-integer match on `win` that #59 built. The one new thing is a
**completion policy**, and it mirrors the Python arm exactly rather than inventing one: a
window is CLOSED when any sender on that chain reports a newer one (`lag=1` in
`TelemClient.windows`). No timer, no barrier, no tolerance.

## 6. The actuation interface -- `set_trim` is not `set_seeds`

Today the fast thread copies the cycle's exact posted dict and substitutes one field,
because "a POST that drops a field ZEROES it at the tracker". That dance is a symptom. Add:

    POST /chord_track/set_trim   [{"prn": 4, "trim_chips": -0.312, "win": 123456}]

* carries **one number**, and it is **absolute, not a delta** -- a dropped message then
  costs latency, not authority;
* writes into the existing `trim[]` vector at the existing Spec-construction point.
  The actuator is **already built and tested**; `code_trim` simply stays false so the
  tracker is a pure actuator and the fleet controller is the only loop;
* **does not touch `seeds[]`, `t_recv`, or `ema_n`.** `set_seeds_callback` resets
  `ema_n = 0` for every seeded PRN. That is harmless today only because `code_trim` is off;
  at 24 Hz it would pin any tracker-side average at warm-up forever. The slow plane keeps
  posting real seeds every cycle so `seed_ttl_s` never expires them.

**Transport.** 6 node hosts x 2 GPUs x 5 chains = 60 endpoint paths, but only **6 hosts**.
Batch per host over a persistent connection: 6 POSTs per frame, ~143/s. Ordinary for C++.
What defeats Python here is not the count -- it is connect-per-POST plus the GIL under five
chain threads.

## 7. Two traps, named before they bite

**(a) `gain` and `leak` are PER-UPDATE, so loop bandwidth scales with rate.** With
`t <- (1-lambda) t + g u` at rate f, the continuous form is
`dt/dt = -lambda*f*t + g*f*u`: steady state (`g*tau/lambda`) is unchanged, but the closed-loop
bandwidth and the noise bandwidth both scale with f. Going 3.1 -> 95.4 Hz multiplies them by
**30x** at unchanged constants. So the seam carries `leak_per_s` and the controller derives
the per-update value from its **achieved** rate. Otherwise the first thing this does is
oscillate, and it will look like the fleet combine's fault.

**(b) One discriminator, one place.** The Python slow DLL must be *retired* for armed PRNs,
not run alongside with its own copy of the tau formula. Same rule as
`gnssSeedTransport.hpp`: the convention lives in one function and both the harness and the
shipped path call it.

⚠️ And the existing actuator gate, `scripts/gnss/fast_trim_e2e.py`, has a **3-PRN fixture**
-- which is precisely why it could not catch the regression where 5 Hz was requested and
1.5 Hz delivered (`fleet_dll_comb` was decoding every PRN and using one). It reached 4.39 Hz
either way. **A gate smaller than production cannot measure a cost that scales with
production.** The gate needs a production-sized PRN and instance count before it is allowed
to certify a rate.

## 8. Milestones

* **F1 -- the observer.** `GnssFleetTrim` consumes `telem_buf`, collates on `win`, computes
  fleet E/P/L/disc/q, serves them. **Posts nothing, actuates nothing.**
  *Gate:* its `disc`/`q` match `combdll.fleet_dll_comb` on the same windows, offline, from
  identical bytes.
* **F2 -- the actuator.** `/chord_track/set_trim` + the batched poster; `code_trim` still
  false. *Gate:* `fast_trim_e2e` at production PRN/instance count sustains >= 90 steps/s and
  >= 23 posts/s, and consecutive posts differ only in `trim_chips`.
* **F3 -- the seam.** `set_policy` / `get_state`; the Python fast-trim thread deleted, the
  Python DLL scoped to unarmed PRNs.
* **F4 -- arm.** gps_l5 armed, the other four chains as control. Judged **paired
  (armed vs control in the same sample), as a duty over hours, on `q`** -- never on
  `deep_snr`, `cn0_coh` or `coh_frac`, which #66 shows fire on noise (a PRN 64 degrees below
  the horizon reads 11.2x its floor).

## 9. Not in scope

Carrier. The carrier trim has its own per-sat residual story (#52, f_carrier) and its own
cadence; folding it in here would couple two loops whose failure modes we cannot yet
separate. Code first, and only once F4 has a duty number.
