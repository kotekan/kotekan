"""Paced garbage collection for the broker process.

The free-threaded interpreter's collector stops every thread and scans the whole heap on each
collection. With every chain in one process the heap holds millions of objects, a collection
takes 0.3-0.9 s, and the allocation rate of thirteen chain passes triggers several a minute.
The telemetry reader is stopped with everything else, so a long collection is a hole in every
chain's telemetry and ends every fleet-ADR arc.

Reference counting frees everything that is not in a cycle, so automatic collection is turned
off and one thread collects on a slow clock instead, logging how long the stop was and how much
cyclic garbage had gathered: that line is the evidence for choosing the interval.
"""
import gc
import threading
import time

from .transport import _log

_lock = threading.Lock()
_thread = None


def _run(interval_s):
    while True:
        time.sleep(interval_s)
        t0 = time.perf_counter()
        n = gc.collect()
        _log(
            "GC: paced collection stopped the process %.2f s, %d unreachable object(s) "
            "after %.0f s" % (time.perf_counter() - t0, n, interval_s)
        )


def start_once(interval_s):
    """Turn automatic collection off and collect every `interval_s` seconds; once per process."""
    global _thread
    if interval_s <= 0:
        return False
    with _lock:
        if _thread is not None:
            return False
        gc.disable()
        _thread = threading.Thread(
            target=_run, args=(float(interval_s),), name="gc-pace", daemon=True
        )
        _thread.start()
    _log("GC: automatic collection off, one paced collection every %.0f s" % interval_s)
    return True
