# Parallelising the broker: the concurrency audit, and the free-threaded plan

**Status 2026-08-24 — INTERPRETER BUILT AND PROVEN OFFLINE; THE FLEET IS STILL ON 3.12.**
Steps 1-3 are done. Free-threaded CPython 3.14.3 is installed at `/home/kvand/gnss/venv-ft`,
all four `broker_equiv` digests and all eight `fleetdll_gate` legs reproduce under it, and the
hot path measures **1.117 s -> 0.307 s** for five chains on cf06 (section 7). What has NOT
happened is the switch: the live broker still runs 3.12 and its 10.04 s cycle.

## 1. Why, as arithmetic

The broker runs five chains as threads of one process. Through 2026-08-23/24 its cycle time
was reduced 15-19 s -> 10.0 s by removing work (see `[[chord-broker-parallelism-ceiling]]`),
and for most of that stretch **cycle time equalled total CPU exactly** — the signature of a
GIL-saturated process. It no longer does:

```
                                      cycle      CPU/cycle-set
before the work-reduction pass        14.06 s       14.38 s     cycle == CPU
after                                 10.00          8.60-8.92  cycle  > CPU
```

Per-chain CPU is now **~1.78 s**. Five chains on five cores would cycle at **~1.8 s**, under
the configured `interval: 2`. Work reduction has stopped converting one-for-one into cycle
time, so that ~5x is the whole remaining win — and it is parallelism, not optimisation.

⚠️ **And it must scale further.** Today's five chains are two bands (GPS L5 + Galileo E5a +
BeiDou B2a at 1176.45 MHz; Galileo E5b + BeiDou B2b at 1207.14). L1 alone will carry GPS
L1C/A + L1C, Galileo E1, BeiDou B1I + B1C and GLONASS L1OF. Any design that parallelises
per BAND is a factor of two today and a bottleneck the moment a band gets crowded.

## 2. Why not processes

`Receiver` — the shared state — is already the right shape for IPC: physics-scoped,
**contribute/consume, never overwrite**, a handful of scalars per chain per cycle. It would
survive being a proxy. Two things would not:

- **The joint filter is per BAND and shared by every chain on it.** Three chains at 1176.45
  MHz feed one `JointReceiverState`, deliberately, so the gauge's 1/N leak shrinks and the
  clock stops being one constellation's private median. Proxying a Kalman-style update across
  processes makes ORDERING a correctness question, not a latency one.
- **The telemetry reader decodes the gather stream once for everyone.** One thread, 46 MB/s,
  ~1.6% CPU, store keyed by chain. N processes means N full decodes, N-1 of them discarded.
  Tolerable at 5; absurd at 15.

Process-splitting also re-introduces exactly what the M6 unification removed
(`docs/CHORD_BROKER_REFACTOR.md`): the receiver clock through a JSON file, the F-engine anchor
fetched twice, BRDC parsed per process.

**Band-per-process remains the fallback** — `broker_multi --only` already exists, so it is a
small change — and it buys 2x today. Keep it in reserve; do not build on it.

## 3. Why free-threading is the fit

The decisive fact: **the broker's only third-party import is numpy** (plus PyYAML in the
launcher). Not astropy, not scipy — verified by importing the module and diffing
`sys.modules`. Both have free-threaded wheels.

The threads already exist. The shared state is already lock-guarded. The reader stays shared,
the joint filter stays in-process memory, and it scales to any number of chains and bands with
no architectural change. It is also the only option in which the natural mental model —
"shared state guarded by a mutex" — is literally true; under the GIL a mutex buys nothing,
because the GIL is what serialises us.

## 4. THE AUDIT: what the GIL is protecting by accident

Every shared object, and whether its locking is complete. "Complete" means following
delegation — an early pass of this audit reported five unlocked readers on `Receiver` that
were one-line delegations to a locked helper. **A concurrency audit that does not follow
delegation manufactures false positives, and a false positive here costs trust in the real
findings.**

| object | shared by | status |
|---|---|---|
| `Receiver` (anchor, BRDC, biases, dr clock, joint) | all chains | ✅ RLock; every reader delegates to the locked `_best` |
| `FleetPublisher` | all chains + the HTTP thread | ✅ locked, including the `do_GET` paths |
| `TelemClient` | reader thread + all chains | ✅ writer and all readers locked; the unlocked residue is single-writer scalars |
| `JointReceiverState` | every chain on a band | ✅ own RLock + `@_locked` on every public method |
| `transport` DNS cache, rate limiter | all chains | ✅ locked |
| log tag, frozen cycle clock | per chain | ✅ `threading.local()` |
| `seed.FIELDS/OWNERS`, `signals._SYS/_CHAINS` | all chains | ✅ read-only after import |
| `seed.SEEN_OWNERS` | all chains | ✅ built-in `set.add` stays atomic under PEP 703 |

⚠️ `JointReceiverState.clk` / `.clk_rate` are deliberately NOT locked, and that is correct:
indices 0 and 1 always exist whatever the membership does, and the comment beside them records
why locking the read accessors was rejected — on 2026-08-15 17:01:29 a diagnostic indexed
`P[0, 57]` on a 57x57 P during a membership resize and the IndexError **killed the gps_l5 chain
thread**, after which its seeds expired and the chain went dark until the next restart. The
read accessors tolerate a stale index on purpose.

### Findings, both fixed

1. **`gnss_ephemeris._atomic_write_bytes` used a fixed `path + ".tmp"`.** Its own docstring
   says the cache is shared across writers, so two of them refreshing the same day's file
   opened the SAME temp path, interleaved bytes into it, and both renamed. `os.replace` makes
   the RENAME atomic and does nothing about two writers sharing the file being renamed.
   Usually the downloads are identical and the corruption is invisible; the case that bites is
   one of them truncated, which publishes a torn gzip every reader then fails to parse.
   **This races today** — file writes release the GIL. Fixed with a per-writer temp name and
   cleanup on failure.
2. **`JointReceiverState.drain_notes` was an unlocked read-then-rebind swap**, so two chains
   draining at once can both take the list and one loses its notes. Diagnostics, which is why
   it went unnoticed: a lost note is silence, and silence is what a healthy filter looks like.
   Now `@_locked` like every other public method.
3. Not a defect, but corrected: `SEEN_OWNERS` justified itself with "atomic under the GIL". The
   conclusion survives free-threading; the reasoning does not, and a stale reason is what gets
   copied to the next global that is not a set.

**The headline is that there was so little to find.** The state was written for concurrent
threads because it already IS concurrent threads — merely GIL-serialised. Free-threading is
close to drop-in.

## 5. The plan

1. **Audit and fix.** ✅ done (this section). Zero behaviour change; all four `broker_equiv`
   digests EQUIVALENT.
2. **Build the interpreter.** ✅ done. No source build was needed and no root: `uv` serves
   python-build-standalone binaries, so this was one command.

   ```
   uv python install 3.14.3+freethreaded
   uv venv --python 3.14.3+freethreaded /home/kvand/gnss/venv-ft
   uv pip install --python /home/kvand/gnss/venv-ft/bin/python 'numpy==2.5.1' PyYAML
   ```

   The interpreter lands in `~/.local/share/uv/python/`, which is on the shared home, so cf06
   and every node see it with nothing to install per host (verified on cf06).

   **numpy is PINNED to 2.5.1 because that is what the 3.12 venv has.** uv resolves 2.5.2 by
   default, and taking it would have varied two axes at once — a moved digest could then have
   been the interpreter or a numpy patch release, with no way to tell them apart. The 3.12
   venv is untouched, so the switch is `GNSS_PY` and the rollback is unsetting it.
3. **Prove the semantics survive.** ✅ done. All four `broker_equiv` digests reproduce
   **byte-identically**, and the eight `fleetdll_gate` legs pass:

   ```
   broker_fake_l5          49f0a392ae1f3f67…   EQUIVALENT
   broker_onsky_l5         8d14c1daa305a91a…   EQUIVALENT
   broker_onsky_e5a        f2f9490f75bec2c6…   EQUIVALENT
   broker_onsky_l5_holds   ed9b24b7e443aa02…   EQUIVALENT
   ```

   Both arms were run in the same session — 3.12 first, to confirm the goldens were green at
   HEAD before asking whether the interpreter moved them. A gate you did not see pass cannot
   tell you what its failure means.

   ⚠️ **This surfaced one real defect**, and it was ours, not 3.14's: `--dr-clock-wait-s`
   quoted a number as `"0.08%"` in its argparse help. Help strings are %-formatted, so
   `--help` had been dead on all 278 flags for as long as that text existed — it fails only
   when help is FORMATTED, so nothing that merely parses args noticed. 3.14 validates
   eagerly in `add_argument`, which turned a broken `--help` into a broker that would not
   start. Fixed in 17e921f34; an AST scan of every `add_argument` in the tree finds no other.
4. **Measure on sky.** ✅ done, all five chains — see section 8. The expectation in this line
   ("cycle -> 2-3 s") was built on a misreading and is struck: the loop was ALREADY paced at
   2 s and the 10 s number was a log throttle.
   `broker_restart.sh` already honours `GNSS_PY`, so this needs no code change:
   `GNSS_PY=/home/kvand/gnss/venv-ft/bin/python scripts/gnss/broker_restart.sh`.
   Pre-switch baseline archived at
   `fixtures/brokerlogs/gnss_broker_20260824_pre_ft.log` (166 cycles/chain).
5. **Then reconsider what is left.** At a 2 s cycle the remaining Python (~0.85 s in
   `coherent_source`, ~0.74 s in `fleet_dll`'s poll) stops being interesting and the C++
   moves can stop.

## 6. ⚠️ The risk none of the gates cover

Free-threading removes the accidental atomicity that a decade of Python habits assume. The
gates are good — four POST-stream digests plus eight `fleetdll_gate` legs — but **not one of
them exercises concurrency**. A race would pass every one. That is the argument for step 1
being an audit rather than a test, and for step 3 being a digest comparison rather than a
soak: a soak over a racy broker returns a plausible number (see `[[no-burn-in-waits]]`).

**That gate now exists: `scripts/gnss/ft_stress.py`.** Three lanes — `Receiver` torn-read,
`JointReceiverState` membership churn (the 2026-08-15 `P[0,57]`-on-a-57x57 shape, driven with
OVERLAPPING PRN sets so threads actually collide on a row), and `drain_notes` accounting. All
three pass on both interpreters at 8 and 16 threads.

The lane that matters is the self-test, because it settles §6's claim by measurement rather
than argument. `--selftest` reinstalls the pre-fix unlocked read-then-rebind and requires the
NOTES lane to catch it:

```
3.12   GIL ON      20 attempts, nothing detected
3.14t  GIL OFF     attempt 1-2, every time  (+2..+29 notes on 3200)
```

**The defect the audit fixed is invisible under the GIL and trips instantly without it.** That
is the whole argument for step 1 being an audit: the gates could not have found it, and a soak
would have returned a healthy-looking number. Non-detection on the GIL arm is reported
INCONCLUSIVE rather than FAIL — it is the expected result there, and a gate that cries red when
it merely could not tell teaches everyone to ignore its red.

⚠️ The observed failure is **duplication, not loss** — two drainers take the same list object
and both return its contents. An operator reads a note twice and concludes the filter acted
twice, which is worse than silence.

⚠️ A pass is evidence, not proof: it runs a fixed number of iterations, not a duration. Run it
again rather than longer.


## 7. What it actually buys, measured

`scripts/gnss/ft_scaling.py` drives the real `fleet_coherent` from N threads on N **distinct**
inputs and compares every thread's result against its own serial run. Distinct inputs matter:
handing every thread the same dict makes agreement trivially true and measures numpy's read
scaling instead of ours — a gate that cannot fail.

On cf06 (the broker host, live fleet running alongside), five chains, production 12x128 shape:

```
                       serial      parallel    speedup
3.12   GIL ON          0.878 s      1.117 s      0.79x
3.14t  GIL OFF         0.722 s      0.307 s      2.35x
```

**The GIL arm is slower threaded than serial.** That is not an anomaly, it is what a
GIL-serialised process does when you add contention and switching to work that was already
serial — and it is the honest baseline, so the number to compare is 1.117 -> 0.307.

The result that matters for §1's warning about crowded bands is not the ratio but the
**flatness**:

```
free-threaded, parallel wall time:   n=1  0.288 s
                                     n=5  0.307 s
                                     n=12 0.300 s
```

Twelve chains cost what one costs. Adding GLONASS L1OF or B1I to a band is then a scheduling
question, not a cycle-time question.

Per-thread free-threading penalty is **~22%** single-threaded (0.214 s vs 0.176 s/chain). It is
paid once, and n=2 already swamps it. On the full replay workload it was not measurable at all
— that path is numpy- and I/O-dominated, and the penalty lands on pure-Python bytecode.

⚠️ **What this does NOT measure.** `fleet_coherent` is pure. The estimator scaling is real, but
the broker's cycle also carries `Receiver`, `JointReceiverState` and `FleetPublisher`, which are
shared and locked — those are covered by the §4 audit, by inspection, and by nothing else. The
speedup on sky will be lower than 3.6x because the cycle is more than this one estimator, and
because 10.0 s > 8.9 s of CPU already: some of the cycle is waiting, and removing the GIL does
not make an HTTP timeout shorter.


## 8. On sky: the A/B/A, and what the headline number actually was

⚠️⚠️ **FIRST, THE CORRECTION.** The "cycle time" this project drove from 15-19 s to 10.0 s is
the spacing of the `fleet DLL [comb]` log line, which is emitted through
`_log_rl(key, msg, every_s=10.0)` — **the log rate limiter's default throttle.** It is not a
cadence and it does not scale with `--interval`. The control loop has been paced at
`--interval` = 2.00 s all along.

`_log_rl` fires on the first pass at or after 10 s and is skipped entirely when the path
produces nothing, so the old 15-19 s was a 10 s floor plus how often the fleet-DLL path came
back empty — **a reliability number wearing a latency number's clothes.** Driving it to 10.0
was a real gain; it saturates there by construction and never measured speed.

Two rules from getting this wrong:
- **A line's period tells you its LOGGER's throttle until you check which logger it uses.**
  `FLEET-INST` is plain `_log` and tracks the loop; `fleet DLL [comb]`, `nav_bits`,
  `narrowed search` and `PRN N: meas` all sit at exactly 10.00 s because they share the default.
- **Don't infer a rate from a log when a counter exists.** `cycle_report()` counts passes.

### The measurement, A/B/A, same code and same instrument on both arms

Busy time per control pass, against the 2.00 s interval:

```
arm            passes   mean busy   max      overran
ft (1)           2697     0.48 s    4.87 s    8  (0.3%)
3.12 GIL         1497     0.85 s    5.06 s   36  (2.4%)
ft (2)           1499     0.49 s    4.54 s    8  (0.5%)
```

Alternated because the sky moves faster than the investigation and the two arms could not be
paired in time. The middle arm is the slow one in **both** directions and the two free-threaded
arms reproduce to 2%, so this is the interpreter, not a trend.

**Free-threading: 0.85 -> 0.485 s per pass (1.75x), overruns 2.4% -> 0.4% (~6x).**

### What that means — and what it does not

The loop fits its interval with ~4x headroom, and **it also fit under 3.12**, at 0.85 s and
2.4% overruns. The work-reduction campaign is what got us inside the window; free-threading
arrived after the goal was met and bought **margin, not a fix**.

The reason to keep it is scaling, not today's five chains. Under the GIL per-chain cost adds
linearly — a sixth and seventh chain walk 0.85 s straight at the interval. Free-threaded, §7
measured parallel wall time FLAT from n=1 to n=12. That is the difference between L1 being a
config change and L1 forcing `--interval` back up.

⚠️ Both arms show a rare tail: max 3.5-5.1 s, ~8 passes in 1500. Unexplained, present on both
interpreters, and therefore not a free-threading artifact. Not chased.
