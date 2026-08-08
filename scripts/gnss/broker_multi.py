#!/usr/bin/env python3
"""One broker process, many signals (task #27 M5).

    broker_multi.py config/gnss_chains_chord.yaml [--only gps_l5] [extra broker args...]

WHAT THIS REPLACES. Today CHORD runs `broker_up.sh` for GPS L5 and `broker_up_extra.sh e5a`
for Galileo E5a, side by side. Those two chains are on the SAME CARRIER -- both pass
`--carrier-hz 1176.45e6` -- through the same feed, the same cable, the same F-engine and
the same GPS-disciplined clock. They are not two receivers; they are one receiver estimated
twice, and everything that costs comes from the process boundary between them:

  * the receiver clock had to be hand-pasted, then serialised through a JSON file with a
    flush cadence and a two-read slew gate, to cross four inches of memory;
  * the F-engine time anchor was fetched and latched independently, so two processes could
    straddle a restart and disagree forever, each certain;
  * the BRDC file was parsed twice and each copy's other constellations thrown away
    (measured: one parse holds 32 GPS and 33 Galileo satellites);
  * the viewer needed one instance per broker.

HOW IT WORKS, AND WHY IT IS SO SMALL. `main()` is a closure whose locals are exactly one
chain's state -- that is what a closure is. So running N chains is calling it N times, in N
threads, with one shared `Receiver`. There is no chain object to build, and the 3252-line
cycle body is untouched.

THREADS, NOT A SEQUENTIAL SWEEP. A chain issues ~25 HTTP requests per cycle against a 2 s
interval with 5 s timeouts; four chains in series would not fit in the interval, and the
first slow endpoint would delay every other signal's seeds. Chains share nothing mutable
except the Receiver, which is locked and contribute/consume only.

A CHAIN THAT DIES MUST NOT TAKE THE OTHERS WITH IT, and must not die quietly either -- a
silently missing chain looks exactly like a satellite-free band. Each thread logs its own
death loudly and the driver reports which chains are still alive on exit.
"""
import argparse
import os
import sys
import threading
import time

K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))

import yaml                                              # noqa: E402
import gps_distributed_broker as broker                  # noqa: E402
from gnss_broker import receiver, transport              # noqa: E402


def flags(d):
    """A {key: value} block -> broker command-line flags.

    Values are passed through as strings, `true` becomes a bare flag and `false`/`null`
    drops it -- so a config reads like the flags it becomes, and a typo'd key surfaces as
    argparse's "unrecognized arguments" rather than being silently ignored.
    """
    out = []
    for k, v in d.items():
        if v is None or v is False:
            continue
        out.append("--" + str(k).replace("_", "-"))
        if v is not True:
            out.append(str(v))
    return out


def load(path):
    cfg = yaml.safe_load(open(path)) or {}
    common = cfg.get("common") or {}
    chains = cfg.get("chains") or {}
    if not chains:
        sys.exit("%s defines no chains" % path)
    out = []
    for name, spec in chains.items():
        merged = dict(common)
        merged.update(spec or {})
        out.append((name, flags(merged)))
    return out


def run_chain(name, argv, rx, alive):
    transport.set_log_tag(name)
    try:
        broker.main(argv, rx=rx)
        transport._log("chain finished (main returned)")
    except transport._TranscriptDone as e:
        # A REPLAY ENDING IS NOT A DEATH. Without this the driver reports "*** CHAIN DIED"
        # on a perfectly successful gate run -- and, worse, prints no digest, so the driver
        # itself cannot be gated against a recording. That is exactly backwards: the piece
        # with the new concurrency is the piece most in need of the gate.
        transport._log("transcript replay complete (%s); %d posts, digest %s"
                       % (e, len(transport._TR.posts), transport._TR.digest()))
        print(transport._TR.digest())
    except SystemExit as e:
        # argparse's ap.error() lands here. In a thread it would otherwise be invisible.
        transport._log("*** CHAIN REFUSED TO START: %s. Its signal is DARK; the others "
                       "keep running." % e)
    except Exception as e:
        import traceback
        transport._log("*** CHAIN DIED: %r\n%s" % (e, traceback.format_exc()))
    finally:
        alive.discard(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--only", action="append", default=[],
                    help="run just these chains (repeatable) -- for bisecting a fleet "
                         "problem down to one signal without editing the config")
    ap.add_argument("--list", action="store_true", help="print the resolved flags and exit")
    a, extra = ap.parse_known_args()

    chains = load(a.config)
    if a.only:
        chains = [c for c in chains if c[0] in a.only]
        if not chains:
            sys.exit("--only %s matched no chain in %s" % (a.only, a.config))

    if a.list:
        for name, argv in chains:
            print("%s:\n  %s\n" % (name, " ".join(argv + extra)))
        return

    rx = receiver.Receiver(log=transport._log)
    transport.set_log_tag("driver")
    transport._log("starting %d chain(s): %s"
                   % (len(chains), ", ".join(n for n, _ in chains)))

    alive = set(n for n, _ in chains)
    threads = []
    for name, argv in chains:
        t = threading.Thread(target=run_chain, args=(name, argv + extra, rx, alive),
                             name=name, daemon=True)
        t.start()
        threads.append(t)
        # Stagger the starts. Not a settling wait -- there is no such thing here (every
        # timescale in this system is seconds or faster). It is so the first chain latches
        # the time anchor and builds the BRDC store while the second is still parsing
        # arguments, which keeps the startup log readable and the shared fetches to one.
        time.sleep(0.25)

    last = None
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)
            now = frozenset(alive)
            if now != last:
                transport.set_log_tag("driver")
                transport._log("chains alive: %s" % (", ".join(sorted(now)) or "NONE"))
                transport._log(rx.summary())
                last = now
    except KeyboardInterrupt:
        transport.set_log_tag("driver")
        transport._log("interrupted; chains are daemon threads and exit with the process")
        return
    transport.set_log_tag("driver")
    transport._log("*** EVERY CHAIN HAS EXITED -- the broker is doing nothing. This is not "
                   "an idle state, it is an outage.")
    sys.exit(1)


if __name__ == "__main__":
    main()
