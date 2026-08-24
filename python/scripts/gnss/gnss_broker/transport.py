"""Broker transport: logging, REST, endpoint expansion, and the transcript gate.

Extracted verbatim from gps_distributed_broker.py (task #27 M1). Nothing here holds any
per-signal or per-chain state, which is exactly why it can move first: the unified broker
needs one copy of it, not one per constellation.
"""
import gzip
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_stages import resolve_stage  # noqa: E402


# ---------------------------------------------------------------------------------------
# DNS CACHE -- the broker's cycle time was name resolution (2026-08-23).
#
# THE MEASUREMENT. `urllib.request.urlopen` resolves the hostname on EVERY call: no
# keep-alive, no connection pool, no cache. fleet_coherent walks its 12 combiner endpoints
# SERIALLY, twice per chain per cycle, so one broker cycle costs ~480 getaddrinfo calls --
# and /etc/resolv.conf carries three search domains, so a short name like "cx19" can cost
# three queries each. On cf06 over 30 s:
#
#     getaddrinfo n=7314   median 0.4 ms   p90 0.7   p99 2.0   max 5005.6 ms
#       >1 s: 6 calls  --  and those 6 ARE ~ALL 30.5 s of DNS time
#
# Six calls in seven thousand hit a 5-SECOND resolver timeout, and they were the entire
# cycle budget: 15 s median, 43 s max, against --interval 2. Live stack sampling put 39.7%
# of all chain-thread wall time in this ONE stdlib frame (76% in transport HTTP overall).
#
# WHY IT TIMES OUT, AND WHY IT IS NOT A DNS FAULT. cf06 resolves via a single server out
# eno8303 -- the one 1 GbE that also carries the fleet's telemetry, measured at 402 Mbps and
# bursty BY DESIGN (the senders are frame-synced, so they burst together). The query packets
# drown in the burst and systemd-resolved falls back to its timeout. CONTROL, same names and
# same server from a quiet host: 69,888 lookups, max 8.0 ms, ZERO over 1 s. So this is the
# mechanism behind the telem-gather comment's "the transport costs ~5 s per cycle, and the
# cost is ON THE WIRE" -- that A-B-A was right; it is DNS loss, not REST bandwidth. All 12
# combiners answer in 0.26 s for 3.9 MB.
#
# ⚠️ THE LOCK IS NEVER HELD ACROSS A LOOKUP. Holding it would serialise every thread's
# resolution behind whichever one is currently stalled for 5 s -- the exact failure this
# exists to remove, made global. Two threads racing to resolve the same name simply both
# resolve it; that is rare and harmless.
#
# ⚠️ STALE-IF-ERROR IS DELIBERATE. `_dns_good` is kept separately from the TTL cache, so a
# lookup that FAILS falls back to the last answer that worked rather than taking the node
# out of the fleet. These are fixed-address lab machines; a stale address is recoverable
# (the connect fails and the poll is skipped for one cycle), a 5 s stall is not.
_DNS_TTL_S = 300.0
_dns_cache = {}          # key -> (expiry, result)
_dns_good = {}           # key -> result, last one that resolved (never expires)
_dns_lock = threading.Lock()
_dns_real = None


def install_dns_cache(ttl_s=None):
    """Wrap socket.getaddrinfo with a TTL cache. Idempotent; safe to call per chain thread.

    Set GNSS_NO_DNS_CACHE=1 to leave the stdlib alone (to reproduce the stalls, or if a name
    ever needs to be re-resolved faster than the TTL).
    """
    global _dns_real, _DNS_TTL_S
    if ttl_s is not None:
        _DNS_TTL_S = float(ttl_s)
    if _dns_real is not None or os.environ.get("GNSS_NO_DNS_CACHE"):
        return
    _dns_real = socket.getaddrinfo

    def _cached(host, port, family=0, type=0, proto=0, flags=0):
        key = (host, port, family, type, proto, flags)
        now = time.time()
        with _dns_lock:
            hit = _dns_cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
        try:
            res = _dns_real(host, port, family, type, proto, flags)
        except Exception:
            with _dns_lock:
                stale = _dns_good.get(key)
            if stale is not None:
                return stale
            raise
        with _dns_lock:
            _dns_cache[key] = (now + _DNS_TTL_S, res)
            _dns_good[key] = res
        return res

    socket.getaddrinfo = _cached


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
        # .gz transparently: an on-sky transcript is ~1.3 MB per cycle (the combiner status
        # responses are the bulk), so a minute of real fleet is ~73 MB raw and ~21 MB
        # compressed. Those live outside git -- the DIGEST is what gets versioned -- and
        # reading them compressed keeps that practical.
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
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
        _t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as h:
                v = json.loads(h.read().decode())
            _http_record("get", url, time.perf_counter() - _t0, True)
        except Exception as e:
            _http_record("get", url, time.perf_counter() - _t0, False)
            if self._mine() and self.mode == "write":
                self._emit({"k": "get", "u": url, "r": None, "e": repr(e)})
            raise
        if self._mine() and self.mode == "write":
            self._emit({"k": "get", "u": url, "r": v})
        return v

    def post(self, url, payload, timeout):
        # The POST stream IS the gate: captured BEFORE any transport can fail, so a run
        # against a dead fleet still produces a comparable trace.
        #
        # ⚠️ ONLY WHILE A TRANSCRIPT IS OPEN. The first version appended unconditionally and
        # nothing ever drains it, so a live broker grew this list forever: MEASURED at
        # 69 MB/hour on the CHORD GPS chain (1.6 GB/day, and it would have multiplied by
        # chain count under broker_multi.py). Nothing in live mode ever reads `posts` --
        # digest() is only called by the harness, which always has a transcript open -- so
        # gating on mode costs the gate nothing.
        if self.mode is not None:
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
        _t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as h:
                s = h.status
            _http_record("post", url, time.perf_counter() - _t0, True)
        except Exception as e:
            _http_record("post", url, time.perf_counter() - _t0, False)
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



# -- HTTP TIMING ----------------------------------------------------------------------
# WHY THIS EXISTS. Free-threading landed (2026-08-24) and the cycle did not move: 10.04 s
# before, 9.98 s after, with the process using 0.72 CORES. Five threads that can now truly
# run at once are using less than one core between them, so the cycle is not compute at all
# -- it is waiting, and no amount of parallelism shortens a wait.
#
# Attribution by log line could only say "~1.7 s elapses here", and said it after a dozen
# UNRELATED lines, which is the signature of one blocking call whose caller varies rather
# than of expensive work. That is as far as the log can take it. This measures the calls
# themselves, in the process that makes them, which is the only place the number is real.
#
# Costs a perf_counter pair and a dict update per request against calls that take ~1 s.
# NOT gated behind a flag: an instrument you have to switch on is one you do not have during
# the incident that needed it.
#
# ⚠️ DIGEST-SAFE BY CONSTRUCTION. It touches neither `posts` nor control flow, so the
# broker_equiv POST stream cannot move -- but re-run the four transcripts anyway, because
# "cannot move" is a claim about code I just wrote.
_http_lk = threading.RLock()
_http = {}          # key -> [n, total_s, max_s, n_fail]


def _http_key(url):
    """host + endpoint, with the instance index folded out.

    gnss0_e5a_inject and gnss1_e5a_inject are the same endpoint on two GPUs and belong in
    one row; cx19 and cx51 are DIFFERENT MACHINES and must not be, because "one node is
    slow" and "this endpoint is slow" call for opposite fixes and the whole point of the
    tool is telling them apart.
    """
    try:
        rest = url.split("://", 1)[1]
        host = rest.split(":", 1)[0].split("/", 1)[0]
        path = rest.split("/", 1)[1] if "/" in rest else ""
    except Exception:
        return url[:64]
    return "%s %s" % (host, re.sub(r"gnss\d+", "gnssN", path))


def _http_record(kind, url, dt, ok):
    k = "%-4s %s" % (kind, _http_key(url))
    with _http_lk:
        e = _http.get(k)
        if e is None:
            _http[k] = [1, dt, dt, 0 if ok else 1]
        else:
            e[0] += 1
            e[1] += dt
            if dt > e[2]:
                e[2] = dt
            if not ok:
                e[3] += 1


_cyc_lk = threading.RLock()
_cyc = {}           # chain -> [n, total_busy_s, max_busy_s, n_overrun]


def record_cycle(chain, busy_s, interval_s):
    """How long one pass of the control loop ACTUALLY took, before its pacing sleep.

    ⚠️ THE LOOP PERIOD IS NOT THE LOOP COST, and reading one for the other cost this project
    real time. `--interval` paces the loop, so once the work fits inside it the observed
    period pins to the interval and stops carrying any information about the work at all --
    it reports the setting back to you. Worse, the headline number everyone was tracking is
    the POLICY cadence at 5x the interval, which pins at 10.00 s and looks like a wall.

    This is the number that actually moves: busy time per pass, and the fraction of passes
    that overran. Slack here is the licence to lower --interval; overruns are the reason it
    was raised.
    """
    with _cyc_lk:
        e = _cyc.get(chain)
        over = 1 if busy_s > interval_s else 0
        if e is None:
            _cyc[chain] = [1, busy_s, busy_s, over]
        else:
            e[0] += 1
            e[1] += busy_s
            if busy_s > e[2]:
                e[2] = busy_s
            e[3] += over


def cycle_report(interval_s=None, reset=True):
    with _cyc_lk:
        snap = sorted(_cyc.items())
        if reset:
            _cyc.clear()
    if not snap:
        return []
    out = ["CYCLE: busy time per control pass (interval %s):"
           % ("?" if interval_s is None else "%.2f s" % interval_s)]
    for c, (n, tot, mx, over) in snap:
        mean = tot / n
        out.append("  %-9s n=%4d  mean %5.2fs  max %5.2fs  overran %d (%.0f%%)%s"
                   % (c, n, mean, mx, over, 100.0 * over / n,
                      "" if interval_s is None
                      else "  slack %.0f%%" % (100.0 * (1.0 - mean / interval_s))))
    return out


def http_timing_report(top=10, reset=True):
    """One block of lines: the endpoints this process actually waited on.

    Sorted by TOTAL time, not by max. A 5 s outlier once a minute is a curiosity; forty
    calls at 1.2 s is the cycle, and sorting by max buries the second behind the first --
    which is how the 2026-08-23 hunt spent an hour on tail latency.
    """
    with _http_lk:
        snap = sorted(_http.items(), key=lambda kv: -kv[1][1])
        wall = sum(v[1] for _, v in snap)
        calls = sum(v[0] for _, v in snap)
        if reset:
            _http.clear()
    if not snap:
        return []
    out = ["HTTP: %d call(s), %.1f s of thread-seconds waiting, top %d by total:"
           % (calls, wall, min(top, len(snap)))]
    for k, (n, tot, mx, nf) in snap[:top]:
        out.append("  %7.2fs %5d call(s) mean %6.3fs max %6.3fs%s  %s"
                   % (tot, n, tot / n, mx, "" if not nf else " FAIL %d" % nf, k))
    return out


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


def log_tag():
    """This thread's chain name ("gps_l5"), or "" outside a chain thread.

    broker_multi runs every chain in one process as a thread and tags it with the chain key
    from gnss_chains_chord.yaml, so this IS the chain identity -- and it is the same string the
    trackers stamp on every telemetry frame (task #59, gnssTelem.hpp). Read it rather than
    re-deriving the chain from --signal: they agree today and nothing enforces that they must.
    """
    return getattr(_tag, "v", "").strip()


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
