"""The interface watcher: the kotekan interface each program the broker talks to serves, read
from the running fleet and compared with the golden this broker speaks (iface.py).

One daemon thread, started by scripts/gnss/broker_multi.py. Its origins are the distinct
host:port of every URL in the chains yaml, each with the roles that set what it is compared on
(iface.ROLES):

    node        trackers, combiner(s), rest-url, spectrum-endpoints (the record assemblers),
                rf-stats-endpoints (the voltage taps) and the nav combiners
    aggregator  detectors
    gather      fleet-trim-url

The signals section is narrowed to the primary and tracker descriptors of the chains that use
the origin (signals._CHAINS).

Each origin's GET /version is read every 60 s, and every 10 s while it is dirty: a read failed,
a mismatch waits for its second read or stands, or a chain's own request to it failed
(transport's error hook). Its document, GET /interfaces/gnss, is fetched only when a section
digest in scope differs from the golden's, and once per document digest. The watcher has its
own urllib client with a 3 s timeout, never transport's, so the broker_equiv transcripts and
the chains' keep-alive pools never see it. It never calls into a chain, and a chain never waits
on it.

    UNKNOWN       never read. A failed read (timeout, refused, an HTTP error, not JSON) keeps
                  whatever verdict the origin had.
    LEGACY        /version carries no gnss interface: a binary built before the registry.
                  Allowed; one WARN.
    UNCOMPARABLE  an error block, another schema, or a document that does not hash to the
                  digests served with it. Allowed; one WARN.
    MATCH         every golden leaf in scope is there and equal, and nothing else is.
    COMPATIBLE    the same, and the binary has more (a new descriptor or constant); the extras
                  are logged once per digest.
    MISMATCH      a golden leaf in scope is missing or different, on two reads at least 10 s
                  apart that give the same digest and the same failing leaves. The first good
                  read ends it.

The mode comes from the chains yaml's common block, `iface: warn` or `iface: {mode: warn,
mode-file: PATH}`, and a mode file (default /var/tmp/gnss-logs/iface_mode, one word) read on
every pass overrides it, so it changes without a restart:

    off      nothing is read and the verdicts freeze: the kill switch.
    warn     (the default) an ERROR on entering MISMATCH, repeated every 10 min, naming the
             failing paths with both values, the node's git_commit_hash and the pin. The
             verdicts are served at the publisher's GET /get_iface, with a summary per chain in
             /get_chains.
    enforce  as warn, and a node origin in MISMATCH is fenced the way a node that is down is
             seen: its REST requests raise transport.IfaceRefused (a URLError) except /version,
             /config, /metrics, /status and /telescope/*; its telemetry frames are dropped
             (TelemClient.iface_dropped); its trackers leave the fleet trim's set_policy
             targets. When only signals leaves fail, only the chains using them are fenced;
             otherwise the whole origin. The gather and the aggregator are singletons: reported,
             never fenced. A failed read never fences anything.

When every node that answers reads MISMATCH with one digest, the broker's own pin is the likely
fault, and an ERROR says so. A watcher that stops passing freezes its verdicts; the driver logs
an alarm once they are 10 min old.

Stdlib only, like iface.py: `python3 -m gnss_broker.iface check-fleet` runs one pass of it.
"""
import threading
import time
import traceback
import urllib.parse

from . import iface, signals, transport

UNKNOWN = "UNKNOWN"
MODES = ("off", "warn", "enforce")
DEFAULT_MODE = "warn"
DEFAULT_MODE_FILE = "/var/tmp/gnss-logs/iface_mode"
POLL_S = 60.0
DIRTY_POLL_S = 10.0
CONFIRM_S = 10.0  # a MISMATCH needs a second read at least this much later
REPEAT_S = 600.0  # a standing MISMATCH's ERROR line, and the stale-verdict alarm
TIMEOUT_S = 3.0
TICK_S = 1.0
# Roles never fenced: one of each serves the whole fleet.
SINGLETONS = ("gather", "aggregator")
# The chains yaml keys whose values are kotekan URLs (broker flags, cli.py), and the role of the
# program behind each. A bare name hangs off rest-url, as in the broker.
URL_ROLES = {
    "rest-url": "node",
    "trackers": "node",
    "combiner": "node",
    "dll-combiners": "node",
    "n2-combiners": "node",
    "spectrum-endpoints": "node",
    "rf-stats-endpoints": "node",
    "cnav-combiner": "node",
    "inav-combiner": "node",
    "fnav-combiner": "node",
    "bcnav1-combiner": "node",
    "bcnav2-combiner": "node",
    "cnav2-combiner": "node",
    "detectors": "aggregator",
    "fleet-trim-url": "gather",
}
REST_URL_DEFAULT = "http://localhost:12048"  # cli.py --rest-url
# Display order, worst first, for a chain's summary.
_WORST = (
    iface.MISMATCH,
    UNKNOWN,
    iface.UNCOMPARABLE,
    iface.LEGACY,
    iface.COMPATIBLE,
    iface.MATCH,
)


def settings(common):
    """(mode, mode file) from a chains yaml's common block: `iface:` absent, a mode word, or
    {mode, mode-file}. ValueError for anything else, so a typo stops the start."""
    v = (common or {}).get("iface")
    mode, mode_file = DEFAULT_MODE, DEFAULT_MODE_FILE
    if isinstance(v, dict):
        extra = set(v) - {"mode", "mode-file", "mode_file"}
        if extra:
            raise ValueError("iface: unknown key(s) %s" % ", ".join(sorted(extra)))
        mode = v.get("mode", mode)
        mode_file = v.get("mode-file", v.get("mode_file", mode_file))
    elif v is not None:
        mode = v
    if mode not in MODES:
        raise ValueError("iface: mode %r is not one of %s" % (mode, ", ".join(MODES)))
    if mode_file is not None and not isinstance(mode_file, str):
        raise ValueError("iface: mode-file %r is not a path" % (mode_file,))
    return mode, mode_file or None


class Origin(object):
    """One host:port and its verdict. Only the watcher's thread touches it."""

    def __init__(self, netloc):
        self.netloc = netloc
        self.base = "http://" + netloc
        self.roles = set()
        self.chains = set()  # every chain with a URL here
        self.node_chains = set()  # those with a node URL here: a chain fence's scope
        self.descriptors = set()  # the signals scope
        self.chain_descriptors = {}  # chain -> its descriptors
        self.state = UNKNOWN
        self.digest = None
        self.commit = None
        self.fails = []
        self.extras = []
        self.reason = ""
        self.since = None
        self.last_try = None
        self.last_read = None
        self.last_error = None
        self.reachable = False
        self.next_poll = 0.0
        self.dirty = False
        # (digest, failing paths, first read time) of a MISMATCH read once
        self.pending = None
        self.fail_key = None  # the failing paths of the standing MISMATCH
        self.warned = set()
        self.extras_logged = set()
        self.error_at = None


def origins_from_chains(cfg):
    """{netloc: Origin} for a parsed chains yaml: every URL in each chain's flags (`common:`
    overlaid with the chain's own block, as broker_multi.load resolves them)."""
    cfg = cfg or {}
    common = cfg.get("common") or {}
    out = {}
    for chain, spec in (cfg.get("chains") or {}).items():
        merged = dict(common)
        merged.update(spec or {})
        kv = {str(k).replace("_", "-"): v for k, v in merged.items()}
        rest = kv.get("rest-url")
        rest = rest if isinstance(rest, str) and rest else REST_URL_DEFAULT
        descs = set(d for d in signals._CHAINS.get(str(kv.get("signal")), ())[:2] if d)
        for key, role in sorted(URL_ROLES.items()):
            v = rest if key == "rest-url" else kv.get(key)
            if v is None or isinstance(v, bool):
                continue
            for url in transport.parse_endpoints(str(v), rest):
                netloc = urllib.parse.urlsplit(url).netloc
                if not netloc:
                    continue
                o = out.get(netloc)
                if o is None:
                    o = out[netloc] = Origin(netloc)
                o.roles.add(role)
                o.chains.add(chain)
                if role == "node":
                    o.node_chains.add(chain)
                if role in ("node", "aggregator"):
                    o.descriptors |= descs
                    o.chain_descriptors.setdefault(chain, set()).update(descs)
    return out


def _fail_rows(fails, limit=None):
    return [
        [iface.path_str(p), iface.show_value(g), iface.show_value(d)]
        for p, g, d in (fails if limit is None else fails[:limit])
    ]


class Watcher(object):
    """The watcher's state and its thread. run_pass() is one pass; start() runs one every
    TICK_S on a daemon thread. Only that thread changes the origins; the publisher reads the
    snapshot and the chain summaries, each replaced whole at the end of a pass."""

    def __init__(
        self,
        origins,
        mode=DEFAULT_MODE,
        mode_file=DEFAULT_MODE_FILE,
        golden=None,
        timeout=TIMEOUT_S,
        log=transport._log,
        set_fence=transport.set_iface_fence,
        confirm=True,
    ):
        self.origins = origins
        self.yaml_mode = mode
        self.mode_file = mode_file
        self.mode = mode
        self.golden = golden if golden is not None else iface.golden()
        self.timeout = timeout
        self.log = log
        self.set_fence = set_fence
        # False for a one-shot report (check-fleet): a MISMATCH stands on its first read.
        self.confirm = confirm
        self.fence = frozenset()
        self.odd_one_out = False
        self._odd_at = None
        self._docs = {}  # document digest -> document, as fetched and checked
        self._dirty = set()
        self._dirty_lock = threading.Lock()
        self._mode_note = None
        self._last_pass = None
        self._started = None
        self._alarm_at = None
        self._snap = {"mode": mode, "origins": {}}
        self._chain_sums = {}
        self._thread = None
        self._stop = threading.Event()

    # -- the thread ------------------------------------------------------------------------
    def start(self):
        """Run passes on a daemon thread, and take transport's request failures as dirt."""
        if self._thread is None:
            self._started = time.time()
            transport.set_iface_error_hook(self.mark_dirty)
            self._thread = threading.Thread(
                target=self._run, name="iface-watch", daemon=True
            )
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        transport.set_log_tag("iface")
        while not self._stop.is_set():
            try:
                self.run_pass()
            except Exception:
                # Never die on one bad pass: the next one may be fine, and a dead watcher
                # freezes every verdict.
                self.log("iface: pass failed\n%s" % traceback.format_exc())
            self._stop.wait(TICK_S)

    def mark_dirty(self, url):
        """transport's error hook: a chain's request to url failed. Cheap and never raises,
        since it runs on the chain's thread."""
        try:
            netloc = urllib.parse.urlsplit(url).netloc
        except Exception:
            return
        if netloc in self.origins:
            with self._dirty_lock:
                self._dirty.add(netloc)

    # -- one pass --------------------------------------------------------------------------
    def run_pass(self, now=None):
        now = time.time() if now is None else now
        self._read_mode()
        with self._dirty_lock:
            dirty, self._dirty = self._dirty, set()
        for n in dirty:
            o = self.origins[n]
            o.dirty = True
            o.next_poll = min(o.next_poll, (o.last_try or 0.0) + DIRTY_POLL_S)
        if self.mode != "off":
            for o in sorted(self.origins.values(), key=lambda o: o.netloc):
                if now < o.next_poll:
                    continue
                o.last_try = now
                try:
                    self._poll(o, now)
                except Exception as e:
                    # One origin's surprise must not stop the others' reads. Logged when it
                    # changes, not on every 10 s retry.
                    err = "watcher error: %r" % e
                    if err != o.last_error:
                        self.log(
                            "iface: %s: %s, verdict stays %s\n%s"
                            % (o.netloc, err, o.state, traceback.format_exc())
                        )
                    o.last_error = err
                    o.dirty = True
                o.next_poll = now + (
                    DIRTY_POLL_S
                    if o.dirty or o.state == iface.MISMATCH or o.pending
                    else POLL_S
                )
            self._repeat_errors(now)
        self._publish_fence()
        self._last_pass = now
        self._snapshot(now)

    def _read_mode(self):
        mode, src, note = self.yaml_mode, "the chains yaml", None
        if self.mode_file:
            word = None
            try:
                with open(self.mode_file) as fh:
                    word = (fh.read().split() or [None])[0]
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as e:
                note = "cannot read %s (%s)" % (self.mode_file, e)
            if word is not None:
                if word.lower() in MODES:
                    mode, src = word.lower(), self.mode_file
                else:
                    note = "%s says %r, not one of %s; ignored" % (
                        self.mode_file,
                        word,
                        ", ".join(MODES),
                    )
        if note is not None and note != self._mode_note:
            self.log("iface: WARN mode file: %s" % note)
        self._mode_note = note
        if mode != self.mode or self._last_pass is None:
            self.log(
                "iface: mode %s (from %s)%s"
                % (
                    mode,
                    src,
                    "" if self._last_pass is None else ", was %s" % self.mode,
                )
            )
        self.mode = mode

    # -- reading one origin ----------------------------------------------------------------
    def _poll(self, o, now):
        try:
            ver = iface.read_version(o.base, self.timeout)
        except iface.SourceError as e:
            return self._read_failed(o, e)
        commit = ver.get("git_commit_hash")
        commit = commit if isinstance(commit, str) else None
        try:
            entry = iface.version_entry(ver, o.base)
            digest = entry.get("digest")
            c = self._compare(o, entry)
        except iface.Legacy as e:
            return self._settle(o, now, iface.LEGACY, commit, reason=str(e))
        except iface.Uncomparable as e:
            return self._settle(o, now, iface.UNCOMPARABLE, commit, reason=str(e))
        except iface.SourceError as e:  # the document could not be fetched
            return self._read_failed(o, e)
        if c.state == iface.MISMATCH:
            return self._mismatch(o, now, commit, digest, c)
        return self._settle(o, now, c.state, commit, digest=digest, c=c)

    def _scope_signals(self, o):
        return sorted(d for d in o.descriptors if d in self.golden.sections["signals"])

    def _compare(self, o, entry):
        """The comparison over o's roles. Equal section digests settle it without fetching the
        document; otherwise it is fetched, once per document digest."""
        sig = self._scope_signals(o)
        paths = iface.scope(sorted(o.roles), sig)
        served = entry.get("sections")
        if isinstance(served, dict) and all(
            served.get(p[0]) == self.golden.section_digests[p[0]] for p in paths
        ):
            return iface.Comparison(iface.MATCH)
        digest = entry.get("digest")
        doc = self._docs.get(digest) if isinstance(digest, str) else None
        if doc is None:
            doc = iface.fetch_doc(o.base, entry, self.timeout)
            if isinstance(digest, str):
                if len(self._docs) >= 16:
                    self._docs.pop(next(iter(self._docs)))
                self._docs[digest] = doc
        return iface.compare(self.golden, doc, role=sorted(o.roles), signals=sig)

    def _read_failed(self, o, e):
        if o.reachable or o.last_error is None:
            self.log(
                "iface: %s unreadable (%s); verdict stays %s" % (o.netloc, e, o.state)
            )
        o.last_error = str(e)
        o.reachable = False
        o.dirty = True

    def _read_ok(self, o, now, commit):
        o.last_read = now
        o.last_error = None
        o.reachable = True
        o.commit = commit

    def _mismatch(self, o, now, commit, digest, c):
        self._read_ok(o, now, commit)
        key = (digest, tuple(iface.path_str(p) for p, _, _ in c.fails))
        if o.state == iface.MISMATCH and (o.digest, o.fail_key) == key:
            o.pending = None
            return
        if self.confirm and (o.pending is None or o.pending[:2] != key):
            o.pending = key + (now,)
            o.dirty = True
            self.log(
                "iface: %s reads a mismatch (%d leaves, document %s); confirming on a read "
                "%d s from now, verdict stays %s"
                % (o.netloc, len(c.fails), digest, CONFIRM_S, o.state)
            )
            return
        if self.confirm and now - o.pending[2] < CONFIRM_S:
            o.dirty = True
            return
        o.pending = None
        o.fail_key = key[1]
        self._enter(o, now, iface.MISMATCH, digest, c.fails, c.extras, "")
        self._mismatch_error(o, now)

    def _settle(self, o, now, state, commit, digest=None, c=None, reason=""):
        self._read_ok(o, now, commit)
        o.pending = None
        o.dirty = False
        fails = c.fails if c is not None else []
        extras = c.extras if c is not None else []
        was = o.state
        self._enter(o, now, state, digest, fails, extras, reason)
        if state in (iface.LEGACY, iface.UNCOMPARABLE) and state not in o.warned:
            o.warned.add(state)
            self.log(
                "iface: WARN %s is %s (kotekan %s): %s. Allowed; it is not checked."
                % (o.netloc, state, commit, reason)
            )
        elif state == iface.COMPATIBLE and digest not in o.extras_logged:
            o.extras_logged.add(digest)
            self.log(
                "iface: %s is COMPATIBLE (kotekan %s, document %s), with %d leaves the "
                "golden lacks: %s"
                % (
                    o.netloc,
                    commit,
                    digest,
                    len(extras),
                    ", ".join(iface.path_str(p) for p in extras[:12])
                    + (" ..." if len(extras) > 12 else ""),
                )
            )
        elif state != was:
            self.log(
                "iface: %s is %s (kotekan %s)%s"
                % (o.netloc, state, commit, ", was %s" % was if was != UNKNOWN else "")
            )

    def _enter(self, o, now, state, digest, fails, extras, reason):
        if state != o.state:
            o.since = now
            o.error_at = None
        o.state = state
        o.digest = digest
        o.fails = list(fails)
        o.extras = list(extras)
        o.reason = reason

    def _mismatch_error(self, o, now):
        o.error_at = now
        self.log(
            "iface: *** ERROR %s MISMATCH (kotekan %s, document %s) against the broker's "
            "golden %s (pin kotekan %s), mode %s: %s"
            % (
                o.netloc,
                o.commit,
                o.digest,
                self.golden.digest,
                self.golden.commit,
                self.mode,
                "; ".join(iface.describe_fails(o.fails, limit=12)),
            )
        )

    def _repeat_errors(self, now):
        for o in sorted(self.origins.values(), key=lambda o: o.netloc):
            if o.state == iface.MISMATCH and now - (o.error_at or 0.0) >= REPEAT_S:
                self._mismatch_error(o, now)
        nodes = [
            o
            for o in self.origins.values()
            if "node" in o.roles and o.reachable and o.state != UNKNOWN
        ]
        odd = (
            len(nodes) >= 2
            and all(o.state == iface.MISMATCH for o in nodes)
            and len(set(o.digest for o in nodes)) == 1
        )
        if odd and (
            not self.odd_one_out
            or self._odd_at is None
            or now - self._odd_at >= REPEAT_S
        ):
            self._odd_at = now
            self.log(
                "iface: *** ERROR broker pin is the odd one out (pin %s): all %d nodes that "
                "answer serve document %s, and it disagrees with the golden %s"
                % (self.golden.commit, len(nodes), nodes[0].digest, self.golden.digest)
            )
        self.odd_one_out = odd

    # -- the fence -------------------------------------------------------------------------
    def _fence_of(self, o):
        """(netloc, chain) pairs fencing o, chain None for every chain; empty unless o is a
        node origin in MISMATCH."""
        if (
            o.state != iface.MISMATCH
            or "node" not in o.roles
            or o.roles & set(SINGLETONS)
        ):
            return set()
        if all(len(p) >= 2 and p[0] == "signals" for p, _, _ in o.fails):
            bad = set(p[1] for p, _, _ in o.fails)
            chains = set(
                ch for ch in o.node_chains if o.chain_descriptors.get(ch, set()) & bad
            )
            if chains:
                return set((o.netloc, ch) for ch in chains)
        return {(o.netloc, None)}

    def _publish_fence(self):
        fence = set()
        if self.mode == "enforce":
            for o in self.origins.values():
                fence |= self._fence_of(o)
        fence = frozenset(fence)
        if fence != self.fence:
            if fence:
                self.log(
                    "iface: ENFORCE fencing %s"
                    % ", ".join(
                        "%s (%s)" % (n, "every chain" if c is None else c)
                        for n, c in sorted(fence, key=lambda x: (x[0], x[1] or ""))
                    )
                )
            else:
                self.log("iface: no origin is fenced")
            self.fence = fence
        self.set_fence(fence)

    # -- what the publisher serves ---------------------------------------------------------
    def _snapshot(self, now):
        fenced = {}
        for n, c in self.fence:
            fenced.setdefault(n, []).append(c)
        origins = {}
        for n, o in sorted(self.origins.items()):
            f = fenced.get(n)
            origins[n] = {
                "roles": sorted(o.roles),
                "chains": sorted(o.chains),
                "state": o.state,
                "since": o.since,
                "kotekan": o.commit,
                "digest": o.digest,
                "fails": _fail_rows(o.fails, 40),
                "n_fails": len(o.fails),
                "n_extras": len(o.extras),
                "reason": o.reason,
                "last_read": o.last_read,
                "last_error": o.last_error,
                "pending": o.pending is not None,
                "fenced": None if f is None else "origin" if None in f else sorted(f),
            }
        sums = {}
        for n, o in self.origins.items():
            for ch in o.chains:
                s = sums.setdefault(
                    ch, {"mode": self.mode, "worst": iface.MATCH, "states": {}}
                )
                s["states"][o.state] = s["states"].get(o.state, 0) + 1
                if _WORST.index(o.state) < _WORST.index(s["worst"]):
                    s["worst"] = o.state
                if any(c is None or c == ch for c in fenced.get(n, ())):
                    s["fenced"] = sorted(s.get("fenced", []) + [n])
        self._chain_sums = sums
        self._snap = {
            "mode": self.mode,
            "golden": self.golden.digest,
            "pin": self.golden.commit,
            "updated": now,
            "odd_one_out": self.odd_one_out,
            "origins": origins,
        }

    def snapshot(self):
        """The verdicts, as GET /get_iface serves them; age_s is the time since the last pass."""
        snap = self._snap
        upd = snap.get("updated")
        return dict(snap, age_s=None if upd is None else round(time.time() - upd, 1))

    def chain_summary(self, chain):
        """A chain's line in /get_chains: the mode, the worst verdict over its origins, the
        count of each, and the origins fenced for it."""
        s = self._chain_sums.get(chain)
        return (
            s if s is not None else {"mode": self.mode, "worst": UNKNOWN, "states": {}}
        )

    def stale_alarm(self, now=None):
        """An alarm line when the last pass (or the start, before any) is over REPEAT_S old,
        at most one per REPEAT_S; else None. For the driver, which outlives a dead watcher
        thread."""
        now = time.time() if now is None else now
        last = self._last_pass if self._last_pass is not None else self._started
        if last is None or now - last < REPEAT_S:
            return None
        if self._alarm_at is not None and now - self._alarm_at < REPEAT_S:
            return None
        self._alarm_at = now
        return (
            "iface: *** the interface watcher's verdicts are %.0f s old: its thread %s"
            % (
                now - last,
                "has died"
                if self._thread is not None and not self._thread.is_alive()
                else "is stuck",
            )
        )


def from_chains(cfg, **kw):
    """A Watcher over a parsed chains yaml, its mode from the common block (settings())."""
    mode, mode_file = settings((cfg or {}).get("common"))
    kw.setdefault("mode", mode)
    kw.setdefault("mode_file", mode_file)
    return Watcher(origins_from_chains(cfg), **kw)


def check_fleet(cfg, timeout=TIMEOUT_S):
    """One read of every origin in a parsed chains yaml, as table lines, and whether any is in
    MISMATCH. A single read decides here: it is a report, not a fence."""
    w = from_chains(
        cfg,
        mode="warn",
        mode_file=None,
        timeout=timeout,
        log=lambda msg: None,
        set_fence=lambda fence: None,
        confirm=False,
    )
    w.run_pass()
    lines = [
        "golden %s, pin kotekan %s" % (w.golden.digest, w.golden.commit),
        "%-22s %-18s %-13s %-12s %s"
        % ("origin", "roles", "verdict", "kotekan", "detail"),
    ]
    bad = False
    for n, o in sorted(w.origins.items()):
        if o.state == iface.MISMATCH:
            bad = True
            detail = "; ".join(iface.describe_fails(o.fails, limit=3))
        elif o.state == iface.COMPATIBLE:
            detail = "%d extra leaves" % len(o.extras)
        elif o.state == UNKNOWN:
            detail = o.last_error or ""
        else:
            detail = o.reason
        lines.append(
            "%-22s %-18s %-13s %-12s %s"
            % (n, ",".join(sorted(o.roles)), o.state, (o.commit or "-")[:12], detail)
        )
    return lines, bad
