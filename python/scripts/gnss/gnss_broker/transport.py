"""Broker transport: logging, REST, endpoint expansion, and the transcript gate.

Holds no per-signal state. One copy serves every chain thread of a broker_multi process; the
log tag, the frozen cycle clock and the keep-alive pool are thread-local.
"""
import gzip
import hashlib
import http.client
import json
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
from datetime import datetime, timezone


# ---------------------------------------------------------------------------------------
# DNS cache. Every new HTTP connection resolves its host name, and the broker host's resolver
# queries share its 1 GbE link with the fleet's frame-synced telemetry bursts, so a lookup
# occasionally loses its packets and waits out the resolver's 5 s timeout, stalling the chain
# thread for all of it.
#
# The lock is never held across a lookup: holding it would serialise every thread's
# resolution behind whichever one is stalled. Two threads resolving the same name both
# resolve it, which is harmless.
#
# Stale-if-error is deliberate: `_dns_good` outlives the TTL cache, so a failed lookup falls
# back to the last answer that worked rather than taking the node out of the fleet. These are
# fixed-address lab machines; a stale address costs a failed connect and one skipped poll,
# a 5 s stall costs the cycle.
_DNS_TTL_S = 300.0
_dns_cache = {}  # key -> (expiry, result)
_dns_good = {}  # key -> result, last one that resolved (never expires)
_dns_lock = threading.Lock()
_dns_real = None


def install_dns_cache():
    """Wrap socket.getaddrinfo with a TTL cache. Idempotent but not locked: the first call
    must precede any concurrent one (broker_multi installs it before any chain starts)."""
    global _dns_real
    if _dns_real is not None:
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
# Keep-alive pool: one TCP connection per (thread, host:port), reused across calls.
#
# The nodes answer in milliseconds, but a fresh connection from the broker host can stall
# for seconds: its 1 GbE ingress carries the fleet's frame-synced telemetry bursts and drops
# some of the nodes' segments, and a lost SYN-ACK has no fast retransmit, only the
# 1 s -> 2 s -> 4 s RTO ladder, where a segment lost on an open connection costs one ~200 ms
# RTO. Reusing connections takes the handshake off that link, and keeps the nodes' TIME-WAIT
# tables from filling with one dead socket per request.
#
# Per thread: http.client connections are not thread-safe and the chain threads run their
# cycles concurrently. A thread-local pool needs no lock and costs threads x nodes sockets;
# _raise_fd_soft_limit lifts the fd soft limit on first use so that is never the failure.
#
# Callers see urlopen's semantics: the same timeout for connect and every read, and
# urllib.error.HTTPError on status >= 400. A failure of a reused connection itself (the
# server closed an idle keep-alive, which is not a fault) gets one retry; a timeout or a
# fresh connection gets none.
_tl = threading.local()
_fd_raised = False


def _raise_fd_soft_limit():
    global _fd_raised
    if _fd_raised:
        return
    _fd_raised = True
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = min(hard, 65536) if hard != resource.RLIM_INFINITY else 65536
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except Exception:
        pass


def _ka_conn(netloc, timeout):
    pool = getattr(_tl, "conns", None)
    if pool is None:
        pool = _tl.conns = {}
        _raise_fd_soft_limit()
    c = pool.get(netloc)
    if c is None:
        host, _, port = netloc.rpartition(":")
        if not host:
            host, port = netloc, ""
        c = http.client.HTTPConnection(host, int(port) if port else 80, timeout=timeout)
        pool[netloc] = c
        return c, False
    c.timeout = timeout
    if c.sock is not None:
        try:
            c.sock.settimeout(timeout)
        except OSError:
            _ka_drop(netloc)
            return _ka_conn(netloc, timeout)
    return c, True


def _ka_drop(netloc):
    pool = getattr(_tl, "conns", None)
    c = pool.pop(netloc, None) if pool else None
    if c is not None:
        try:
            c.close()
        except Exception:
            pass


_RETRY_ON = (
    http.client.RemoteDisconnected,
    http.client.BadStatusLine,
    http.client.CannotSendRequest,
    http.client.ResponseNotReady,
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
)


def http_request(method, url, timeout, data=None, headers=None):
    """(status, body bytes) over a pooled keep-alive connection; HTTPError on status >= 400."""
    u = urllib.parse.urlsplit(url)
    path = u.path or "/"
    if u.query:
        path += "?" + u.query
    for attempt in (0, 1):
        c, reused = _ka_conn(u.netloc, timeout)
        try:
            c.request(method, path, body=data, headers=headers or {})
            r = c.getresponse()
            body = r.read()
        except _RETRY_ON:
            _ka_drop(u.netloc)
            if reused and attempt == 0:
                continue
            raise
        except Exception:
            _ka_drop(u.netloc)
            raise
        if r.will_close:
            _ka_drop(u.netloc)
        if r.status >= 400:
            raise urllib.error.HTTPError(url, r.status, r.reason, r.headers, None)
        return r.status, body


# ---------------------------------------------------------------------------------------
# Transcript: record/replay of every external interaction, the broker's refactoring gate.
# One run's whole conversation with the world is recorded, replayed against changed code
# (scripts/gnss/broker_equiv.py), and the POST stream must come out byte-identical. That bar
# is legitimate here: Python neither contracts nor reassociates float expressions, so a pure
# code move reproduces arithmetic exactly and a digest mismatch means a real change.
#
# Three independent streams (`now`, `get`, `post`) rather than one interleaved log: a
# refactor may move when the clock is read relative to a poll, but must not change which
# endpoints are polled, in what order, or what is posted. A get out of order fails on its URL.
#
# The cycle clock (`_now()`) is frozen for the duration of a cycle, so every gate in one pass
# evaluates at one instant instead of at whenever the slow endpoints before it returned.
# The main loop's pacing sleep reads the real clock, so the control cadence does not depend
# on it.
# ---------------------------------------------------------------------------------------


class _TranscriptDone(Exception):
    """Replay reached the end of the recording -- a normal, successful termination."""


class _Transcript:
    def __init__(self):
        self.mode = None  # None (live) | "write" | "read"
        self._fh = None
        self._rd = {"now": [], "get": [], "post": []}
        self._ix = {"now": 0, "get": 0, "post": 0}
        self._owner = None  # only the thread that opened the transcript is transcribed
        # The frozen clock is per thread, not per process: one process runs several chains in
        # threads, each on its own cycle, and a shared clock would stamp chain A's cycle with
        # chain B's instant, which looks like clock jitter rather than a data race.
        self._t = threading.local()
        self.posts = []  # replay+record: the ordered POST stream (the gate output)
        self.argv = None  # the recording run's own argv, carried in the header

    # -- lifecycle ------------------------------------------------------------------
    def open_write(self, path, argv):
        self.mode, self._owner = "write", threading.get_ident()
        # Line-buffered: a recording normally ends with the broker being killed, and a block
        # buffer would lose the last few seconds or leave a half-written JSON line that makes
        # the whole transcript unreadable.
        self._fh = open(path, "w", buffering=1)
        # The header carries the recording run's argv so a replay is self-describing:
        # broker_equiv.py re-invokes the broker with the same flags instead of retyped ones,
        # several of which would fail silently if mistyped.
        self._emit({"k": "argv", "v": list(argv)})

    def open_read(self, path):
        self.mode, self._owner = "read", threading.get_ident()
        # Reads .gz transparently: on-sky transcripts are large (mostly combiner status
        # responses) and are kept gzipped outside git; only their digests are versioned
        # (scripts/gnss/fixtures).
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
        # reads are not control flow and must not enter the stream, or the recording would
        # depend on when a browser happened to poll.
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
        # The frozen clock applies in every mode, including live. Freezing it only while a
        # transcript is open would make a recording describe a run that never happens in
        # production, and the gate would measure the instrumentation.
        #
        # The discriminator is whether this thread runs cycles. A chain thread calls tick()
        # and gets its own frozen instant; the publisher's HTTP thread never does, so `v` is
        # unset there and it reads the real clock, which is right since its timestamps are not
        # part of any cycle.
        v = getattr(self._t, "v", None)
        return time.time() if v is None else v

    # -- transport ------------------------------------------------------------------
    def get(self, url, timeout):
        if self.mode == "read" and self._mine():
            r = self._take("get")
            if r["u"] != url:
                raise RuntimeError(
                    "TRANSCRIPT DIVERGENCE at get #%d: recorded %s, replay "
                    "asked for %s" % (self._ix["get"] - 1, r["u"], url)
                )
            if r.get("e"):
                raise RuntimeError(r["e"])
            return r["r"]
        _t0 = time.perf_counter()
        try:
            v = json.loads(http_request("GET", url, timeout)[1].decode())
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
        # The POST stream is the gate's output, captured before any transport can fail, so a
        # run against a dead fleet still produces a comparable trace.
        #
        # Only while a transcript is open: nothing drains `posts`, so a live broker would grow
        # it without bound, and digest() is read only at the end of a replay.
        if self.mode is not None:
            self.posts.append((url, json.dumps(payload, sort_keys=True)))
        if self.mode == "read" and self._mine():
            r = self._take("post")
            if r["u"] != url:
                raise RuntimeError(
                    "TRANSCRIPT DIVERGENCE at post #%d: recorded %s, replay "
                    "sent to %s" % (self._ix["post"] - 1, r["u"], url)
                )
            if r.get("e"):
                raise RuntimeError(r["e"])
            return r["s"]
        data = json.dumps(payload).encode()
        _t0 = time.perf_counter()
        try:
            s = http_request(
                "POST",
                url,
                timeout,
                data=data,
                headers={"Content-Type": "application/json"},
            )[0]
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


# -- HTTP timing ----------------------------------------------------------------------
# Per-endpoint call count, total and max wait, and failures, measured in the process that
# makes the calls (http_timing_report). A log line can show only where time passed, not
# which call it was spent waiting on.
#
# Always on: it costs a perf_counter pair and a dict update per request, and an instrument
# that must be switched on is missing during the incident that needs it. It touches neither
# `posts` nor control flow, so the transcript digest cannot move.
_http_lk = threading.RLock()
_http = {}  # key -> [n, total_s, max_s, n_fail]


def _http_key(url):
    """host + endpoint, with the instance index folded out.

    gnss0_e5a_inject and gnss1_e5a_inject are the same endpoint on two GPUs and belong in
    one row; cx19 and cx51 are different machines and must not share one, because "one node
    is slow" and "this endpoint is slow" call for different fixes.
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
_cyc = {}  # chain -> [n, total_busy_s, max_busy_s, n_overrun]


def record_cycle(chain, busy_s, interval_s):
    """Record how long one pass of the control loop took, before its pacing sleep.

    The loop period is not the loop cost: --interval paces the loop, so once the work fits
    the observed period pins to the interval and says nothing about the work. Nor is the
    spacing of a log line a cadence: a line logged through _log_rl repeats at its throttle
    (every_s, 10 s by default) whatever the interval, and not at all on a pass that logs
    nothing. So this counts and times the passes themselves: busy time per pass and the
    fraction that overran. Slack here is room to lower --interval; overruns are a reason to
    raise it.
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
    out = [
        "CYCLE: busy time per control pass (interval %s):"
        % ("?" if interval_s is None else "%.2f s" % interval_s)
    ]
    for c, (n, tot, mx, over) in snap:
        mean = tot / n
        out.append(
            "  %-9s n=%4d  mean %5.2fs  max %5.2fs  overran %d (%.0f%%)%s"
            % (
                c,
                n,
                mean,
                mx,
                over,
                100.0 * over / n,
                ""
                if interval_s is None
                else "  slack %.0f%%" % (100.0 * (1.0 - mean / interval_s)),
            )
        )
    return out


def http_timing_report(top=10, reset=True):
    """One block of lines: the endpoints this process actually waited on.

    Sorted by total time, not by max: a 5 s outlier once a minute matters less than forty
    calls at 1.2 s, and sorting by max would bury the second behind the first.
    """
    with _http_lk:
        snap = sorted(_http.items(), key=lambda kv: -kv[1][1])
        wall = sum(v[1] for _, v in snap)
        calls = sum(v[0] for _, v in snap)
        if reset:
            _http.clear()
    if not snap:
        return []
    out = [
        "HTTP: %d call(s), %.1f s of thread-seconds waiting, top %d by total:"
        % (calls, wall, min(top, len(snap)))
    ]
    for k, (n, tot, mx, nf) in snap[:top]:
        out.append(
            "  %7.2fs %5d call(s) mean %6.3fs max %6.3fs%s  %s"
            % (tot, n, tot / n, mx, "" if not nf else " FAIL %d" % nf, k)
        )
    return out


_TR = _Transcript()


def _now():
    """The frozen cycle clock. See the _Transcript note: one cycle, one instant."""
    return _TR.now()


def axis_now(ctx):
    """"Now" as a capture age (s since sample 0) on the F-engine hop axis: the time every seed
    label (ref_hop), model epoch and detection age is referred to, so no absolute wall-clock
    reading reaches a radiometric number (#208). Under --dr-fengine-axis it is the hop counter
    interpolated by wall ELAPSED seconds only: the filtered hop-minus-wall offset (fe_off) plus
    this cycle's frozen clock, or the raw newest hop plus the seconds since it was polled while
    the filter has no sample yet. Without the flag, or before the first telemetry, it is the
    frozen clock minus utc0_sample0, the only form there is then; the two agree to NTP plus the
    telemetry lag. None until utc0_sample0 (the F-engine's time of sample 0) is known."""
    if not getattr(ctx, "utc0_sample0", 0.0):
        return None
    now_w = _now()
    fe_axis = getattr(ctx, "fe_axis", [None])[0]
    if getattr(ctx.args, "dr_fengine_axis", False) and fe_axis is not None:
        fe_off = getattr(ctx, "fe_off", [None])[0]
        if fe_off is not None:
            return fe_off + now_w
        _feh, _few = fe_axis
        return _feh / ctx.args.hops_per_sec + (now_w - _few)
    return now_w - ctx.utc0_sample0  # 208-allow: no axis yet


def axis_now_utc(ctx):
    """axis_now placed on UTC through utc0_sample0, as an aware datetime; None without an
    axis."""
    t = axis_now(ctx)
    if t is None:
        return None
    return datetime.fromtimestamp(ctx.utc0_sample0 + t, tz=timezone.utc)


def _get(url, timeout=5.0):
    return _TR.get(url, timeout)


def _post(url, payload, timeout=5.0):
    return _TR.post(url, payload, timeout)


# Per-chain log tag. One process runs several chains in threads and their log lines
# interleave, and a line such as "DLL: PRN 3 disc +0.012" means something different on L5
# than on E5a. Set per thread (set_log_tag), so a chain tags itself once and every _log on
# that thread carries it.
#
# Empty by default, so a single-chain run's lines keep their format: the equivalence gate
# covers POSTs, not stderr, and a reformatted log breaks existing greps.
_tag = threading.local()


def set_log_tag(tag):
    _tag.v = (" " + tag) if tag else ""


def log_tag():
    """This thread's chain name ("gps_l5") on a broker_multi chain thread; "" when unset.

    broker_multi runs every chain in one process as a thread and tags it with the chain key
    from its chains yaml (config/gnss_chains_chord.yaml), so this is the chain identity, and
    it is the same string the trackers stamp on every telemetry frame (gnssTelem.hpp). Read it
    rather than re-deriving the chain from --signal: nothing enforces that the two agree.
    """
    return getattr(_tag, "v", "").strip()


def _log(msg):
    # Wall-clock time to the millisecond, so events can be timed from the log alone:
    # cheap, greppable, sortable.
    print(
        "[broker%s %s] %s"
        % (getattr(_tag, "v", ""), datetime.now().strftime("%H:%M:%S.%f")[:-3], msg),
        file=sys.stderr,
        flush=True,
    )


# Rate-limit keys include the chain's log tag: two chains sharing one key would silence each
# other's lines at random, which reads as "that chain stopped logging".
_log_rl_last = {}


def _log_rl(key, msg, every_s=10.0):
    """Rate-limited _log for per-cycle state lines: at most one line per (chain tag, key)
    per every_s seconds of the cycle clock (_now). Per-cycle lines repeat mostly unchanged,
    so this keeps the journal readable while keeping the history dense enough to reconstruct
    events. Event lines go through _log and are never limited."""
    now = _now()
    key = (getattr(_tag, "v", ""), key)  # see _log_rl_last: per-chain, not shared
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
        out.append(tok[: m.start()] + str(i).zfill(width) + tok[m.end() :])
    res = []
    for o in out:  # handle any further ranges in the same token
        res.extend(expand_token(o))
    return res


def resolve_prefix(entry, default_base):
    """Endpoint prefix (everything before /<verb>) for a list entry.

    Absolute http(s) entries are used as-is; bare names hang off --rest-url.
    """
    entry = entry.strip()
    if entry.startswith("http://") or entry.startswith("https://"):
        return entry.rstrip("/")
    return default_base.rstrip("/") + "/" + entry.strip("/")


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
