# Silent kotekan kill — forensic harness (armed 2026-07-28)

Kotekan has vanished twice with **no crash line, no core, not OOM, GPU healthy** — once after
~2 h up, once ~6 s after a healthy startup. See buglist 2026-07-28. This harness captures the
NEXT death without perturbing the real-time pipeline (no gdb pre-attached, which could mask a
timing bug and adds ptrace overhead).

## What is armed

1. **bpftrace death-trap** — `diag/../../../../tmp/gnss_forensics/death_trap.bt`, running as
   root, logging to `/tmp/gnss_forensics/death_trap.log`. Restart-proof (filters on
   `comm == "kotekan"`, not a pid). It records:
   - **KILL-SYSCALL / TGKILL**: any SIGKILL(9)/SIGABRT(6) to any pid, with the **sender**
     pid/comm/uid. If kotekan dies and a line here names a sender → an external process killed
     it (a script, a stray `pkill -9`, a supervisor, me). SIGTERM is excluded (it triggers
     kotekan's clean, LOGGED shutdown — not the silent signature).
   - **SIGNAL->kotekan**: every fatal signal (4/6/7/8/9/11) actually delivered to a kotekan
     process. The distinguisher:
     | trap shows | means |
     |---|---|
     | SEGV/ABRT/BUS/ILL/FPE, no KILL-SYSCALL | kotekan crashed ITSELF → read the core |
     | KILL(9), no KILL-SYSCALL | OOM / kernel-generated (but we ruled OOM out — 74 GB free) |
     | KILL(9) WITH a KILL-SYSCALL | the named external process did it |
   Validated 2026-07-28: a test `kill -9` was caught with the sender named.

2. **Core dumps** — `kernel.core_pattern` is set (sysctl, out-of-band) to
   `/tmp/gnss_cores/core.%e.%p.sig%s.%t`, bypassing apport (which discards cores for
   non-packaged binaries). `run_3band.sh` now does `ulimit -c unlimited` before launching
   kotekan, so every future launch can write a core. A self-crash → a **symbolizable** core
   (the binary is not stripped: 60,866 text symbols; function names, no line numbers without a
   `-g` rebuild). Postmortem: `gdb <binary> /tmp/gnss_cores/core.* -batch -ex 'thread apply all bt'`.

## Reading the evidence after a death

```
cat /tmp/gnss_forensics/death_trap.log          # the signal + sender
ls -lt /tmp/gnss_cores/                          # a core => self-crash
gdb $(readlink /proc/<any-live-pid>/exe or the kotekan binary) /tmp/gnss_cores/core.kotekan.* \
    -batch -ex 'thread apply all bt' 2>&1 | head -80
```

## Re-arming / persistence

- The bpftrace trap is **restart-proof** (comm-filtered) and survives kotekan restarts. It does
  NOT survive a host reboot or being killed — re-launch with:
  `sudo setsid bash -c 'exec bpftrace /tmp/gnss_forensics/death_trap.bt > /tmp/gnss_forensics/death_trap.log 2>&1' &`
- `core_pattern` is a sysctl → resets to apport on host reboot. Re-set with:
  `sudo sysctl -w kernel.core_pattern='/tmp/gnss_cores/core.%e.%p.sig%s.%t'` (and `mkdir -p
  /tmp/gnss_cores && chmod 777` it). Original saved in `/tmp/gnss_forensics/orig_core_pattern.txt`.
- `ulimit -c unlimited` in run_3band.sh makes cores automatic per launch (committed).

## Leading hypothesis (unproven)

Both deaths correlated with the UNIFIED viewer + an actively-polling browser; the unified feed
polled all three `/adcstat` (the single-libevent-thread wedge endpoint) concurrently. Removed
(round-robin) and the unified deploy is held. If the node now dies with only the FLAT viewer
and no heavy polling, the viewer is exonerated and the trap output points elsewhere.
