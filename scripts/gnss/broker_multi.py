#!/usr/bin/env python3
"""One broker process, many signals: every chain of a chains yaml runs as a thread.

    broker_multi.py config/gnss_chains_chord.yaml [--list] [extra broker args...]

A chain's flags are the yaml's `common:` block overlaid with its own block under `chains:`;
the extra args are appended to every chain. --list prints the resolved flags and exits.

Chains on one carrier (GPS L5 and Galileo E5a both run at --carrier-hz 1176.45e6) see the
same feed, cable, F-engine and GPS-disciplined clock: they are one receiver, estimated per
signal. In one process they share a `Receiver` (gnss_broker/receiver.py): one F-engine time
anchor, one multi-system BRDC store, and clock estimates one chain contributes and another
consumes. They also share one REST publisher.

`broker.main()` keeps exactly one chain's state in its locals, so running N chains is
calling it N times, in N threads, with one shared Receiver; there is no chain object.

Threads, not a sequential sweep: a chain makes many HTTP requests per cycle with 5 s
timeouts against a 2 s interval, so chains in series would not fit, and one slow endpoint
would delay every other signal's seeds. Chains share nothing mutable except the Receiver
(locked, contribute/consume only) and the publisher.

A chain that dies must not take the others with it, and must not die quietly: a missing
chain looks exactly like a satellite-free band. Each thread logs its own death, the driver
logs the set of live chains whenever it changes, and the process exits 1 when none is left.
"""
import argparse
import faulthandler
import os
import signal
import sys
import threading
import time

# SIGUSR1 dumps every thread's stack to stderr (the log), to answer which chain thread is
# waiting on what when a cycle runs long: `wchan` says only futex, and py-spy cannot attach
# (ptrace_scope is 1 on these hosts and the daemonised broker is nobody's child).
#
#     kill -USR1 $(pgrep -f '[b]roker_multi')      -> every thread's stack to stderr (the log)
#
# One dump at a time; never drive it as a sampling profiler. faulthandler walks the other
# threads' frames from a signal handler, without the GIL, while they run and mutate them, and
# rapid repeated dumps against busy chain threads can kill the process with no traceback. To
# profile, profile a replay (broker_equiv.py) or launch the broker as py-spy's child.
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))
sys.path.insert(0, os.path.join(K, "config"))

import gnss_fleet  # noqa: E402
import gps_distributed_broker as broker  # noqa: E402
from gnss_broker import iface, iface_watch, publish, receiver, transport  # noqa: E402


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


def read(path):
    """The chains yaml, parsed, its `fleet` endpoint keys expanded from the fleet list
    (config/gnss_fleet.py)."""
    try:
        return gnss_fleet.load_chains(path)
    except gnss_fleet.FleetError as e:
        sys.exit("%s: %s" % (path, e))


def load(path, cfg=None):
    cfg = read(path) if cfg is None else cfg
    common = dict(cfg.get("common") or {})
    # The interface watcher's settings (gnss_broker/iface_watch.py), not a chain flag.
    common.pop("iface", None)
    chains = cfg.get("chains") or {}
    if not chains:
        sys.exit("%s defines no chains" % path)
    out = []
    for name, spec in chains.items():
        merged = dict(common)
        merged.update(spec or {})
        out.append((name, flags(merged)))
    return out


def run_chain(name, argv, rx, alive, pub=None):
    transport.set_log_tag(name)
    try:
        broker.main(argv, rx=rx, publisher=pub)
        transport._log("chain finished (main returned)")
    except SystemExit as e:
        # argparse's ap.error() lands here. In a thread it would otherwise be invisible.
        transport._log(
            "*** CHAIN REFUSED TO START: %s. Its signal is dark; the others "
            "keep running." % e
        )
    except Exception as e:
        import traceback

        transport._log("*** CHAIN DIED: %r\n%s" % (e, traceback.format_exc()))
    finally:
        alive.discard(name)


def _slack_interval(chains, extra):
    """The slack column's interval: the --interval every chain that sets one agrees on, else
    None, since a wrong denominator is worse than none. A chain runs argv + extra and the
    broker's parser keeps the last --interval, so the last one there is the chain's."""
    ivs = set()
    for _, argv in chains:
        args, iv = argv + extra, None
        for i, a in enumerate(args):
            if a == "--interval" and i + 1 < len(args):
                iv = args[i + 1]
            elif a.startswith("--interval="):
                iv = a.split("=", 1)[1]
        if iv is not None:
            try:
                ivs.add(float(iv))
            except ValueError:
                pass
    return ivs.pop() if len(ivs) == 1 else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config")
    ap.add_argument(
        "--list",
        action="store_true",
        help="print each chain's resolved broker flags (extra args included) and exit",
    )
    a, extra = ap.parse_known_args()

    cfg = read(a.config)
    chains = load(a.config, cfg)

    if a.list:
        for name, argv in chains:
            print("%s:\n  %s\n" % (name, " ".join(argv + extra)))
        return

    rx = receiver.Receiver(log=transport._log)
    transport.set_log_tag("driver")
    # The kotekan interface this broker speaks. telem loads it on import, so a missing or
    # corrupt golden has already stopped the start, never a chain mid-run.
    golden = iface.golden()
    transport._log("iface: golden %s, pin kotekan %s" % (golden.digest, golden.commit))
    # The watcher compares what every origin in the chains yaml serves with the golden, on a
    # thread of its own (gnss_broker/iface_watch.py). A bad `iface:` setting stops the start.
    try:
        watcher = iface_watch.from_chains(cfg)
    except ValueError as e:
        sys.exit("%s: %s" % (a.config, e))
    # One publisher for the whole process, on the first non-zero --publish-port in chain
    # order, so one viewer origin serves every chain. Chains are told apart by
    # /<chain>/get_status or ?chain=<id>.
    port = None
    for name, argv in chains:
        if "--publish-port" in argv:
            p = argv[argv.index("--publish-port") + 1]
            if p not in ("0", ""):
                port = int(p)
                break
    # Before any chain starts polling: an uncached DNS lookup can stall a chain thread for
    # the resolver's 5 s timeout (see transport.install_dns_cache).
    transport.install_dns_cache()
    pub = publish.FleetPublisher(port, transport._log) if port else None
    if pub:
        transport._log("fleet publisher shared by every chain on :%d" % port)
        pub.iface = watcher
    transport._log(
        "iface: watching %d origin(s), mode %s, mode file %s"
        % (len(watcher.origins), watcher.yaml_mode, watcher.mode_file)
    )
    watcher.start()
    transport._log(
        "starting %d chain(s): %s" % (len(chains), ", ".join(n for n, _ in chains))
    )

    alive = set(n for n, _ in chains)
    threads = []
    for name, argv in chains:
        t = threading.Thread(
            target=run_chain,
            args=(name, argv + extra, rx, alive, pub),
            name=name,
            daemon=True,
        )
        t.start()
        threads.append(t)
        # Stagger the starts, not a settling wait: the first chain usually latches the time
        # anchor and builds the BRDC store before the next asks. Both run outside the
        # Receiver's lock, so chains that miss together would each fetch; the stagger also
        # keeps the startup log readable.
        time.sleep(0.25)

    # Cycle and HTTP timing reports. The driver is the only thread that belongs to no chain,
    # so it logs the process-wide numbers. Every 60 s, not every cycle: each report is about
    # ten lines and what it measures moves on the scale of minutes. Both reports reset on read,
    # so each block covers the interval since the last.
    interval = _slack_interval(chains, extra)
    last, t_http = None, time.time()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)
            now = frozenset(alive)
            if now != last:
                transport.set_log_tag("driver")
                transport._log("chains alive: %s" % (", ".join(sorted(now)) or "NONE"))
                transport._log(rx.summary())
                last = now
            if time.time() - t_http >= 60.0:
                t_http = time.time()
                transport.set_log_tag("driver")
                for ln in transport.cycle_report(interval_s=interval):
                    transport._log(ln)
                for ln in transport.http_timing_report(top=12):
                    transport._log(ln)
                alarm = watcher.stale_alarm()
                if alarm:
                    transport._log(alarm)
    except KeyboardInterrupt:
        transport.set_log_tag("driver")
        transport._log(
            "interrupted; chains are daemon threads and exit with the process"
        )
        return
    transport.set_log_tag("driver")
    transport._log(
        "*** EVERY CHAIN HAS EXITED -- the broker is doing nothing. This is not "
        "an idle state, it is an outage."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
