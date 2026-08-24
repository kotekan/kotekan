# Parallelising the broker: the concurrency audit, and the free-threaded plan

**Status 2026-08-24 — AUDIT DONE, NOTHING SWITCHED.** Two real defects fixed (both
pre-existing, neither free-threading-specific); the rest of the shared state was already
built for concurrent threads. No interpreter has been installed and no behaviour has changed.

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
2. **Build the interpreter.** Ubuntu 24.04 ships 3.12, so 3.13t/3.14t must come from
   deadsnakes or a source build into `/home/kvand` (no root needed). A venv with free-threaded
   numpy + PyYAML. **Infrastructure, not code, and reversible: the 3.12 venv stays.**
3. **Prove the semantics survive.** Run the EXISTING broker under the free-threaded
   interpreter against the recorded transcripts. If all four digests reproduce, the arithmetic
   and the POST stream are unchanged by the interpreter. This is the cheapest possible
   go/no-go and it needs no fleet.
4. **Measure one band on sky**, then all five chains. Expect cycle -> ~2-3 s.
5. **Then reconsider what is left.** At a 2 s cycle the remaining Python (~0.85 s in
   `coherent_source`, ~0.74 s in `fleet_dll`'s poll) stops being interesting and the C++
   moves can stop.

## 6. ⚠️ The risk none of the gates cover

Free-threading removes the accidental atomicity that a decade of Python habits assume. The
gates are good — four POST-stream digests plus eight `fleetdll_gate` legs — but **not one of
them exercises concurrency**. A race would pass every one. That is the argument for step 1
being an audit rather than a test, and for step 3 being a digest comparison rather than a
soak: a soak over a racy broker returns a plausible number (see `[[no-burn-in-waits]]`).

If a concurrency gate is wanted later, the shape is a stress harness that drives `Receiver`
and `JointReceiverState` from N threads with a deterministic seed and asserts the invariants
(gauge median zero, membership consistency, no lost contributions) — not a longer run of the
broker.
