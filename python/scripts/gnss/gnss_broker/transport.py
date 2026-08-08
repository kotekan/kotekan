"""Broker transport: logging, REST, endpoint expansion, and the transcript gate.

Extracted verbatim from gps_distributed_broker.py (task #27 M1). Nothing here holds any
per-signal or per-chain state, which is exactly why it can move first: the unified broker
needs one copy of it, not one per constellation.
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_stages import resolve_stage  # noqa: E402


# ---------------------------------------------------------------------------------------
# TRANSCRIPT -- record/replay of every external interaction (task #27 M0, the refactor gate).
#
# WHY. This file is 6400 lines, `main()` is 5100 of them, and every comment block in it is a
# scar from a real outage. It cannot be restructured on inspection. The gate is behavioural
# equivalence: record one run's entire conversation with the world, replay it against the
# refactored code, and require the POST stream to come out BYTE-IDENTICAL.
#
# Byte-identical is a legitimate bar here, unlike in the CUDA kernels: Python neither
# contracts nor reassociates float expressions, so a pure code move reproduces arithmetic
# exactly. A hash mismatch therefore means a real change, never numerical weather.
#
# THREE INDEPENDENT STREAMS (`now`, `get`, `post`) rather than one interleaved log. A
# refactor legitimately reorders WHEN the clock is read relative to a poll; it must not
# change WHICH endpoints are polled, in what order, or what is posted. Separate streams
# make the first free and keep the second two strict -- a get that arrives out of order
# fails on the URL, which is exactly the signal wanted.
#
# ONE DELIBERATE BEHAVIOUR CHANGE, and it is the reason `_now()` exists at all: the cycle
# clock is FROZEN for the duration of a cycle. Until now the ~40 clock reads in one pass
# drifted apart by however long the pass took, so a gate evaluated late in the cycle used a
# different "now" than one evaluated early.
#
# MEASURED, do not guess (the first draft of this comment said "~0.3 s" from nothing):
# cf06's live GPS L5 broker, 600 log lines, intra-cycle spread median 0.035 s, p90 1.71 s,
# max 1.79 s -- against a 2 s interval. So on the real fleet the spread is most of a cycle,
# not a rounding error. It is still far inside every threshold these gates use (10 s log
# limits, 60 s anchor re-check, minutes for staleness and watchdogs), and it was never
# meaningful ordering -- it was whichever endpoints happened to be slow that pass. Freezing
# makes one cycle evaluate at one instant, which is what all of them meant.
#
# The loop's own sleep computation deliberately keeps a REAL clock read (below), so the
# control cadence is unchanged.
# ---------------------------------------------------------------------------------------

class _TranscriptDone(Exception):
    """Replay reached the end of the recording -- a normal, successful termination."""


class _Transcript:
    def __init__(self):
        self.mode = None          # None (live) | "write" | "read"
        self._fh = None
        self._rd = {"now": [], "get": [], "post": []}
        self._ix = {"now": 0, "get": 0, "post": 0}
        self._owner = None        # only the main thread is transcribed; see below
        # PER THREAD, not per process (task #27 M5). One process now runs several chains
        # in several threads, each on its own cycle; a shared frozen clock would have chain
        # B silently stamping chain A's cycle with B's instant -- and the failure would look
        # like clock jitter, not like a data race.
        self._t = threading.local()
        self.posts = []           # replay+record: the ordered POST stream (the gate output)
        self.argv = None          # the recording run's own argv, carried in the header

    # -- lifecycle ------------------------------------------------------------------
    def open_write(self, path, argv):
        self.mode, self._owner = "write", threading.get_ident()
        # LINE-buffered: a recording is normally ended by killing the broker, and a block
        # buffer would throw away the last few seconds -- or, worse, leave a half-written
        # JSON line that makes the whole transcript unreadable.
        self._fh = open(path, "w", buffering=1)
        # The header carries the recording run's argv so a replay is self-describing: the
        # harness re-invokes the broker with the SAME flags rather than a human retyping 30
        # of them and quietly getting one wrong. (broker_up_extra.sh is 12 hand-typed
        # constants, several of which fail silently when mistyped -- that class of error is
        # exactly what a gate must not depend on.)
        self._emit({"k": "argv", "v": list(argv)})

    def open_read(self, path):
        self.mode, self._owner = "read", threading.get_ident()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["k"] == "argv":
                    self.argv = r["v"]
                else:
                    self._rd[r["k"]].append(r)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def _mine(self):
        # The publisher serves HTTP from a daemon thread and reads the clock there. Those
        # reads are not part of the control flow and must not enter the stream, or the
        # recording would depend on when a browser happened to poll.
        return self.mode is not None and threading.get_ident() == self._owner

    def _emit(self, rec):
        self._fh.write(json.dumps(rec) + "\n")

    def _take(self, kind):
        i = self._ix[kind]
        if i >= len(self._rd[kind]):
            raise _TranscriptDone("%s stream exhausted after %d entries" % (kind, i))
        self._ix[kind] = i + 1
        return self._rd[kind][i]

    # -- clock ----------------------------------------------------------------------
    def tick(self):
        """Start a new cycle: sample (or replay) the frozen clock."""
        if self.mode == "read":
            self._t.v = self._take("now")["v"]
        else:
            self._t.v = time.time()
            if self._mine():
                self._emit({"k": "now", "v": self._t.v})
        return self._t.v

    def now(self):
        # ⚠️ THE FROZEN CLOCK APPLIES IN EVERY MODE, INCLUDING LIVE. The first version of
        # this gated on `_mine()`, i.e. on a transcript being open -- so production kept a
        # live clock and only a RECORDING froze it. That is the one thing a recorder must
        # never do: the transcript would then describe a run that never happens, and the
        # gate would be measuring the instrumentation. Caught 2026-08-08 while making the
        # clock thread-local for M5.
        #
        # The discriminator is not "is a transcript open" but "is this a thread that runs
        # cycles". A chain thread calls tick() and gets its own frozen instant; the
        # publisher's HTTP thread never does, so `v` is unset there and it reads the real
        # clock -- which is right, since its timestamps are not part of any cycle.
        v = getattr(self._t, "v", None)
        return time.time() if v is None else v

    # -- transport ------------------------------------------------------------------
    def get(self, url, timeout):
        if self.mode == "read" and self._mine():
            r = self._take("get")
            if r["u"] != url:
                raise RuntimeError("TRANSCRIPT DIVERGENCE at get #%d: recorded %s, replay "
                                   "asked for %s" % (self._ix["get"] - 1, r["u"], url))
            if r.get("e"):
                raise RuntimeError(r["e"])
            return r["r"]
        try:
            with urllib.request.urlopen(url, timeout=timeout) as h:
                v = json.loads(h.read().decode())
        except Exception as e:
            if self._mine() and self.mode == "write":
                self._emit({"k": "get", "u": url, "r": None, "e": repr(e)})
            raise
        if self._mine() and self.mode == "write":
            self._emit({"k": "get", "u": url, "r": v})
        return v

    def post(self, url, payload, timeout):
        # The POST stream IS the gate: capture it in every mode, before any transport can
        # fail, so a run against a dead fleet still produces a comparable trace.
        self.posts.append((url, json.dumps(payload, sort_keys=True)))
        if self.mode == "read" and self._mine():
            r = self._take("post")
            if r["u"] != url:
                raise RuntimeError("TRANSCRIPT DIVERGENCE at post #%d: recorded %s, replay "
                                   "sent to %s" % (self._ix["post"] - 1, r["u"], url))
            if r.get("e"):
                raise RuntimeError(r["e"])
            return r["s"]
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as h:
                s = h.status
        except Exception as e:
            if self._mine() and self.mode == "write":
                self._emit({"k": "post", "u": url, "s": None, "e": repr(e)})
            raise
        if self._mine() and self.mode == "write":
            self._emit({"k": "post", "u": url, "s": s})
        return s

    def digest(self):
        """Hash of the ordered POST stream -- the equivalence gate's single number."""
        h = hashlib.sha256()
        for url, body in self.posts:
            h.update(url.encode())
            h.update(b"\x00")
            h.update(body.encode())
            h.update(b"\n")
        return h.hexdigest()


_TR = _Transcript()


def _now():
    """The frozen cycle clock. See the _Transcript note: one cycle, one instant."""
    return _TR.now()


def _get(url, timeout=5.0):
    return _TR.get(url, timeout)


def _post(url, payload, timeout=5.0):
    return _TR.post(url, payload, timeout)


# PER-CHAIN LOG TAG (task #27 M5). One process now runs several chains in several threads,
# and their log lines interleave. Without a tag the journal becomes unreadable at exactly
# the moment it matters most -- and worse, plausibly misreadable: "DLL: PRN 3 disc +0.012"
# means something different on L5 than on E5a, and nothing in the line says which. Set per
# thread, so a chain tags itself once and every _log below it inherits it.
#
# Empty by default, so a single-chain run's output is byte-identical to before -- the
# equivalence gate covers POSTs, not stderr, and a silently reformatted log is exactly the
# kind of change that makes an old autopsy grep stop matching.
_tag = threading.local()


def set_log_tag(tag):
    _tag.v = (" " + tag) if tag else ""


def _log(msg):
    # Timestamped (2026-07-19): every autopsy this week had to reconstruct event times by
    # correlating line numbers against the status stream -- the 07-18 carrier-latch hunt
    # lost an hour to it. Wall-clock, subsecond: cheap, greppable, sortable.
    print("[broker%s %s] %s" % (getattr(_tag, "v", ""),
                                datetime.now().strftime("%H:%M:%S.%f")[:-3], msg),
          file=sys.stderr, flush=True)


# Rate-limit keys are per THREAD as well: two chains sharing one key would silence each
# other's lines at random, which reads as "that chain stopped logging".
_log_rl_last = {}


def _log_rl(key, msg, every_s=10.0):
    """Rate-limited _log for PER-CYCLE state lines (meas/pred, cp-fit, clock, active...):
    at the 0.2 s poll cadence they were ~25-30 MB/h per broker (~5 GB/day fleet-wide,
    measured 2026-07-19) while carrying ~50x duplicate content. One line per key per
    every_s keeps the journal readable and the history dense enough for every autopsy
    this project has actually run (the 07-18 hunts used >=1 s granularity). EVENT lines
    (HOLD/RELEASE/ESCAPE/REACQ/WATCHDOG/TRANSLATE/fits-changed...) stay unlimited."""
    now = _now()
    key = (getattr(_tag, "v", ""), key)   # see _log_rl_last: per-chain, not shared
    if now - _log_rl_last.get(key, 0.0) >= every_s:
        _log_rl_last[key] = now
        _log(msg)


def expand_token(tok):
    """Expand the first bash-style {a..b} range in a token, recursing for more.

    Zero-pads to the operand width iff either operand is written zero-padded
    (e.g. {00..49} -> 00..49, but {0..49} -> 0..49), matching shell brace ranges.
    """
    m = re.search(r"\{(\d+)\.\.(\d+)\}", tok)
    if not m:
        return [tok]
    lo, hi = m.group(1), m.group(2)
    padded = (len(lo) > 1 and lo[0] == "0") or (len(hi) > 1 and hi[0] == "0")
    width = max(len(lo), len(hi)) if padded else 0
    a, b = int(lo), int(hi)
    step = 1 if b >= a else -1
    out = []
    for i in range(a, b + step, step):
        out.append(tok[:m.start()] + str(i).zfill(width) + tok[m.end():])
    res = []
    for o in out:  # handle any further ranges in the same token
        res.extend(expand_token(o))
    return res


def resolve_prefix(entry, default_base):
    """Endpoint prefix (everything before /<verb>) for a list entry.

    Absolute http(s) entries are used as-is; bare names hang off --rest-url. Bare names
    are ALIAS-RESOLVED against the live pipeline (gnss_stages): the tri-constellation
    configs name the GPS chain gps_search/gps_track/gps_combiner, matching gal_*/bds_*,
    while the older single-constellation benches still use search/track/combiner -- either
    spelling works against either config.
    """
    entry = entry.strip()
    if entry.startswith("http://") or entry.startswith("https://"):
        return entry.rstrip("/")
    return default_base.rstrip("/") + "/" + resolve_stage(default_base,
                                                          entry.strip("/"))


def parse_endpoints(csv, default_base):
    """Comma list -> resolved endpoint prefixes, with {a..b} ranges expanded."""
    prefixes = []
    for raw in csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        for tok in expand_token(raw):
            prefixes.append(resolve_prefix(tok, default_base))
    return prefixes
