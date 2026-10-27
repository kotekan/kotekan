"""Which fleet nodes answer, and what the broker does when they stop and come back.

Every chain posts seeds to every tracker instance each cycle, and before this a node that had
stopped answering was simply called again: each call cost a refused connect (or, for a host
gone from the network, a full timeout), and logged a line (10-26: ~60,000 `set_seeds ...
Connection refused` lines in 47 minutes for two stopped nodes). This module holds per-node
state, keyed by REST origin (host:port), shared by every chain in the process:

  UP -> DOWN  after at least down_min_fails connection-level failures spanning down_after_s
              with no success between. Only failures that mean "nothing answered" count:
              refused, reset, timeout, unreachable, DNS. An HTTP error means the node is up.
              Logged once.
  DOWN        every request to the origin raises transport.NodeDown at once, without a
              network attempt, so each caller's unreachable path applies at no cost. One
              daemon thread probes each DOWN origin's /telescope/time0_ns every probe_s.
  DOWN -> UP  on a probe that answers. Logged with the outage's length, and with the node's
              frame 0 against the anchor the broker latched: a node that comes back on a
              different frame 0 is an F-engine re-base (the anchor check then restarts the
              broker; this line says why the node was gone).

Fleet return (the 10-01 rule, runbook s.5): when every registered node has been DOWN for at
least return_restart_s, the broker's dead-reckoned state (held code phases, fits, holds) has
gone stale while nothing measured it, and the cure has been a broker restart once the nodes
are back. restart_due() turns True once the fleet returns: half the registered nodes UP, or
return_settle_s after the first. The chain loop then exits the process (code
EXIT_FLEET_RETURNED) for systemd to relaunch, unless a boresight transit is in the freeze cone
(restart_decision), in which case it waits for it to clear. A fleet that goes dark again
before the restart fires cancels it, and the dark period resumes from its original start: a
node whose REST server answers once and then stops (cx44, 10-26 19:31) is not a return.

Off (no hooks, no thread) under a transcript: a replay must see the same requests in the same
order, and a probe thread would talk to the live world.

@author Keith Vanderlinde
"""

import errno
import socket
import threading
import time
import urllib.error
import urllib.parse

from gnss_broker import transport
from gnss_broker.transport import _log

#: Process exit code for a fleet-return restart (gps_distributed_broker's anchor change is 3).
EXIT_FLEET_RETURNED = 4

_CONN_ERRNOS = (
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
    errno.EHOSTDOWN,
    errno.ETIMEDOUT,
)


def is_conn_failure(e):
    """True when `e` means nothing answered at the origin (refused, reset, timeout,
    unreachable, DNS), False for an answer (an HTTP error) or anything else."""
    if isinstance(e, urllib.error.HTTPError):
        return False
    if isinstance(e, urllib.error.URLError) and isinstance(e.reason, BaseException):
        e = e.reason
    if isinstance(e, (ConnectionError, TimeoutError, socket.timeout, socket.gaierror)):
        return True
    return isinstance(e, OSError) and e.errno in _CONN_ERRNOS


class NodeHealth(object):
    """The per-origin state machine. `clock` and `probe_get` are injectable for tests:
    probe_get(netloc) returns the node's time0_ns (int) or raises."""

    def __init__(
        self,
        down_after_s=10.0,
        down_min_fails=3,
        probe_s=15.0,
        return_restart_s=1200.0,
        return_settle_s=180.0,
        clock=time.time,
        probe_get=None,
        log=_log,
    ):
        self.down_after_s = float(down_after_s)
        self.down_min_fails = int(down_min_fails)
        self.probe_s = float(probe_s)
        self.return_restart_s = float(return_restart_s)
        self.return_settle_s = float(return_settle_s)
        self.clock = clock
        self.probe_get = probe_get or _probe_time0
        self.log = log
        self._lk = threading.Lock()
        self._n = {}  # netloc -> state dict
        self.anchor_s = None  # the broker's latched frame 0, UTC seconds (note_anchor)
        self.dark_since = None  # every registered node DOWN since
        self.return_t = None  # first node back after a long dark
        self.dark_from = None  # start of the dark period return_t ended
        self.restart_due_since = None
        self._thread = None

    # -- registration -----------------------------------------------------------------
    def register(self, bases):
        """Track these node origins (URLs or host:port)."""
        with self._lk:
            for b in bases:
                n = urllib.parse.urlsplit(b).netloc if "://" in b else b
                if n and n not in self._n:
                    self._n[n] = {
                        "up": True,
                        "fails": 0,
                        "fail_since": None,
                        "down_t": None,
                        "err": None,
                        "probe_t": None,
                    }

    def note_anchor(self, time0_s):
        """The frame 0 the broker latched (UTC seconds), for the UP line's comparison."""
        self.anchor_s = float(time0_s) if time0_s else None

    # -- transport hooks --------------------------------------------------------------
    def gate(self, netloc):
        st = self._n.get(netloc)
        return st is not None and not st["up"]

    def result(self, netloc, ok, exc=None):
        st = self._n.get(netloc)
        if st is None:
            return
        if ok or not is_conn_failure(exc):
            # an answer, even an HTTP error, says the node is up
            with self._lk:
                st["fails"], st["fail_since"] = 0, None
            return
        now = self.clock()
        with self._lk:
            if not st["up"]:
                return
            st["fails"] += 1
            st["err"] = "%s" % (exc,)
            if st["fail_since"] is None:
                st["fail_since"] = now
            if (
                st["fails"] >= self.down_min_fails
                and now - st["fail_since"] >= self.down_after_s
            ):
                st["up"], st["down_t"], st["probe_t"] = False, now, now
                n_fails, since = st["fails"], st["fail_since"]
                down = True
            else:
                down = False
        if down:
            self.log(
                "NODE DOWN %s: %d connection failures over %.0f s, none answered (%s) -- "
                "requests to it now fail at once; probing /telescope/time0_ns every %.0f s"
                % (netloc, n_fails, now - since, st["err"], self.probe_s)
            )
            self._fleet_changed(now)

    # -- probing ----------------------------------------------------------------------
    def probe_once(self):
        """Probe every DOWN origin whose probe is due; bring back the ones that answer."""
        now = self.clock()
        with self._lk:
            due = [
                n
                for n, st in self._n.items()
                if not st["up"] and now - (st["probe_t"] or 0.0) >= self.probe_s
            ]
            for n in due:
                self._n[n]["probe_t"] = now
        for n in due:
            try:
                t0 = self.probe_get(n)
            except Exception:
                continue
            with self._lk:
                st = self._n[n]
                gone = now - (st["down_t"] or now)
                st.update(up=True, fails=0, fail_since=None, err=None)
            # the anchor check's own tolerance (gps_distributed_broker: 1 ms)
            if self.anchor_s is None or not t0:
                anchor = "no anchor to compare"
            elif abs(t0 / 1e9 - self.anchor_s) <= 1e-3:
                anchor = "frame 0 unchanged"
            else:
                anchor = (
                    "frame 0 CHANGED (%+.3f s): an F-engine re-base; the anchor check "
                    "restarts the broker" % (t0 / 1e9 - self.anchor_s)
                )
            self.log("NODE UP %s after %.0f s DOWN; %s" % (n, gone, anchor))
            self._fleet_changed(now)

    def _fleet_changed(self, now):
        with self._lk:
            n_up = sum(1 for st in self._n.values() if st["up"])
            n_all = len(self._n)
            cancelled = None
            if n_up == 0 and self.dark_since is None:
                if self.return_t is not None:
                    # dark again before the restart fired: not a return; the dark period
                    # resumes from where it began
                    self.dark_since = cancelled = self.dark_from
                    self.return_t = self.dark_from = None
                else:
                    self.dark_since = now
                dark_start = True
            else:
                dark_start = False
            if n_up > 0 and self.dark_since is not None:
                dark_s = now - self.dark_since
                if self.return_restart_s > 0.0 and dark_s >= self.return_restart_s:
                    self.return_t, self.dark_from = now, self.dark_since
                    returned = dark_s
                else:
                    returned = None
                self.dark_since = None
            else:
                returned = None
        if cancelled is not None:
            self.log(
                "FLEET DARK again: none of the %d registered node(s) answers; the pending "
                "fleet-return restart is cancelled and the dark period runs on from %.0f s ago"
                % (n_all, now - cancelled)
            )
        elif dark_start:
            self.log(
                "FLEET DARK: none of the %d registered node(s) answers -- seeds and polls "
                "go nowhere; the dead-reckoned state ages from here" % n_all
            )
        if returned is not None:
            self.log(
                "FLEET RETURNING after %.0f s dark (>= %.0f s): the dead-reckoned state is "
                "stale, so the broker restarts once half the nodes are back or %.0f s pass, "
                "outside a boresight transit (10-01 rule)"
                % (returned, self.return_restart_s, self.return_settle_s)
            )

    # -- the fleet-return restart -----------------------------------------------------
    def restart_due(self, now=None):
        """True once a fleet that was dark for return_restart_s has returned."""
        if self.return_t is None:
            return False
        now = self.clock() if now is None else now
        with self._lk:
            n_up = sum(1 for st in self._n.values() if st["up"])
            n_all = len(self._n)
        return 2 * n_up >= n_all or now - self.return_t >= self.return_settle_s

    def restart_decision(self, near, freeze_deg, now=None):
        """'exit', 'defer' (due, but a satellite is inside the transit freeze cone: a
        restart there bootstraps the clock from noise) or None (not due)."""
        if not self.restart_due(now):
            return None
        if near is not None and freeze_deg and near[0] < freeze_deg:
            return "defer"
        return "exit"

    def start(self):
        """The probe thread, once."""
        with self._lk:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run, name="node-health", daemon=True
            )
            self._thread.start()

    def _run(self):
        transport.set_log_tag("nodes")
        ev = threading.Event()
        step = max(1.0, min(5.0, self.probe_s / 3.0))
        while True:
            ev.wait(step)
            try:
                self.probe_once()
            except Exception as e:  # never let the prober die silently
                self.log("node-health probe: %s" % e)


def _probe_time0(netloc, timeout=2.0):
    """GET http://netloc/telescope/time0_ns straight through http_request: not through the
    transcript, the gate (the node is DOWN, which the gate would refuse) or the timing table."""
    import json

    body = transport.http_request(
        "GET", "http://%s/telescope/time0_ns" % netloc, timeout
    )[1]
    return int(json.loads(body.decode()).get("time0_ns", 0))


# The process's tracker (enable). None until a chain enables it, and under a transcript.
TRACKER = None
_enable_lk = threading.Lock()


def enable(bases, **kw):
    """Create the process's tracker on first call (later chains add their bases), install the
    transport hooks and start the prober. Returns the tracker."""
    global TRACKER
    with _enable_lk:
        if TRACKER is None:
            TRACKER = NodeHealth(**kw)
            transport.set_node_hooks(TRACKER.gate, TRACKER.result)
            TRACKER.start()
        TRACKER.register(bases)
        return TRACKER
