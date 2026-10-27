"""Node health: DOWN after connection failures, requests then refused at once, probed back UP,
and the fleet-return restart.

    python3 -m gnss_broker.test_nodehealth     (from python/scripts/gnss)

@author Keith Vanderlinde
"""

import errno
import socket
import urllib.error

from gnss_broker import nodehealth, sky, transport
from gnss_broker.nodehealth import NodeHealth, is_conn_failure

_fails = []
A, B, C, D = "cx47:12048", "cx42:12048", "cx19:12048", "cx27:12048"


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class Clock(object):
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def tracker(nodes, **kw):
    clk, logs = Clock(), []
    answers = {}  # netloc -> time0_ns, or absent for a node that still does not answer

    def probe(n):
        if n not in answers:
            raise ConnectionRefusedError(errno.ECONNREFUSED, "refused")
        return answers[n]

    h = NodeHealth(clock=clk, probe_get=probe, log=logs.append, **kw)
    h.register(nodes)
    return h, clk, logs, answers


def refused():
    return ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")


def test_classify():
    print("what counts as nothing answering")
    check(is_conn_failure(refused()), "refused")
    check(is_conn_failure(socket.timeout("timed out")), "timeout")
    check(is_conn_failure(socket.gaierror(-2, "Name or service not known")), "DNS")
    check(is_conn_failure(OSError(errno.EHOSTUNREACH, "No route")), "host unreachable")
    check(
        is_conn_failure(urllib.error.URLError(refused())), "a URLError wrapping refused"
    )
    check(
        not is_conn_failure(urllib.error.HTTPError("u", 500, "x", None, None)),
        "an HTTP error is an answer",
    )
    check(not is_conn_failure(ValueError("bad json")), "a bad body is not")


def test_down_and_gate():
    print("UP -> DOWN")
    h, clk, logs, _ = tracker([A, B])
    for dt in (0.0, 2.0):
        clk.t += dt
        h.result(A, False, refused())
    check(not h.gate(A), "two failures over 2 s: still UP")
    clk.t += 9.0
    h.result(A, False, refused())
    check(
        h.gate(A) and not h.gate(B),
        "the third, 11 s after the first: DOWN; B unaffected",
    )
    check(any(m.startswith("NODE DOWN cx47") for m in logs), "logged once as NODE DOWN")
    n = len(logs)
    h.result(A, False, refused())
    check(len(logs) == n, "further failures while DOWN log nothing")
    h2, clk2, _l, _a = tracker([A])
    h2.result(A, False, refused())
    clk2.t += 20.0
    h2.result(A, False, urllib.error.HTTPError("u", 404, "x", None, None))
    clk2.t += 1.0
    h2.result(A, False, refused())
    h2.result(A, False, refused())
    check(
        not h2.gate(A),
        "an HTTP error between failures resets the run (the node answered)",
    )
    check(not h2.gate("cx99:12048"), "an unregistered origin is never gated")


def test_transport_gate():
    print("the transport refuses a DOWN origin without a network attempt")
    h, clk, _logs, _ = tracker([A])
    calls = []
    real = transport.http_request

    def fake(method, url, timeout, data=None, headers=None):
        calls.append(url)
        clk.t += 5.0
        raise refused()

    transport.http_request = fake
    transport.set_node_hooks(h.gate, h.result)
    try:
        for _ in range(3):
            try:
                transport._TR.post("http://%s/gnss0_inject/set_seeds" % A, {}, 1.0)
            except ConnectionRefusedError:
                pass
        check(len(calls) == 3 and h.gate(A), "three refused posts, 10 s apart: DOWN")
        try:
            transport._TR.get("http://%s/buffers" % A, 1.0)
            got = None
        except transport.NodeDown as e:
            got = e
        check(
            got is not None and len(calls) == 3,
            "the next get raises NodeDown and never reaches the network",
        )
        check(
            isinstance(got, urllib.error.URLError),
            "NodeDown is a URLError, so every caller's unreachable path applies",
        )
    finally:
        transport.http_request = real
        transport.set_node_hooks(None, None)


def test_probe_up():
    print("DOWN -> UP on a probe")
    h, clk, logs, answers = tracker([A], probe_s=15.0)
    h.note_anchor(1792947270.000003890)
    for _ in range(3):
        h.result(A, False, refused())
        clk.t += 5.0
    check(h.gate(A), "DOWN")
    clk.t += 15.0
    h.probe_once()
    check(h.gate(A), "a probe that is refused leaves it DOWN")
    answers[A] = 1792947270000003890
    h.probe_once()
    check(h.gate(A), "the next probe waits probe_s")
    clk.t += 15.0
    h.probe_once()
    up = [m for m in logs if m.startswith("NODE UP cx47")]
    check(
        not h.gate(A) and up and "frame 0 unchanged" in up[0], "UP, frame 0 unchanged"
    )
    for _ in range(3):
        h.result(A, False, refused())
        clk.t += 5.0
    answers[A] = 1793000000000000000
    clk.t += 15.0
    h.probe_once()
    up = [m for m in logs if m.startswith("NODE UP cx47")]
    check(
        "frame 0 CHANGED" in up[-1], "a node back on a new frame 0 is called a re-base"
    )


def test_up_kicks_the_boresight():
    print("NODE UP re-reads the pointing at once")
    ups = []
    h, clk, _logs, answers = tracker([A, B], on_up=ups.append)
    for _ in range(3):
        h.result(A, False, refused())
        clk.t += 5.0
    answers[A] = 1
    clk.t += 15.0
    h.probe_once()
    check(ups == [A], "on_up is called with the origin that came back")
    sky._bore_kick.clear()
    sky.kick_boresight(A)
    check(sky._bore_kick.is_set(), "kick_boresight wakes the watch thread")
    sky._bore_kick.clear()


def _all_down(h, clk, nodes):
    for _ in range(3):
        for n in nodes:
            h.result(n, False, refused())
        clk.t += 5.0


def test_fleet_return():
    print("the fleet-return restart (the 10-01 rule)")
    nodes = [A, B, C, D]
    h, clk, logs, answers = tracker(
        nodes, return_restart_s=1200.0, return_settle_s=180.0
    )
    _all_down(h, clk, nodes)
    check(any(m.startswith("FLEET DARK") for m in logs), "every node DOWN: FLEET DARK")
    clk.t += 1300.0
    answers[A] = 1
    h.probe_once()
    check(
        any(m.startswith("FLEET RETURNING") for m in logs) and not h.restart_due(),
        "one of four back after 1300 s dark: returning, not yet due",
    )
    answers[B] = 1
    clk.t += 16.0
    h.probe_once()
    check(h.restart_due(), "half the fleet back: due")
    check(h.restart_decision(None, 5.0) == "exit", "no transit: exit")
    check(h.restart_decision((2.4, ("E", 34)), 5.0) == "defer", "E34 at 2.4 deg: defer")
    check(h.restart_decision((7.0, ("G", 3)), 5.0) == "exit", "nearest 7 deg: exit")
    h2, clk2, _l2, ans2 = tracker(nodes)
    _all_down(h2, clk2, nodes)
    clk2.t += 1300.0
    ans2[A] = 1
    h2.probe_once()
    clk2.t += 179.0
    check(not h2.restart_due(), "one node back: not due before return_settle_s")
    clk2.t += 2.0
    check(h2.restart_due(), "...due after it")
    h3, clk3, _l3, ans3 = tracker(nodes)
    _all_down(h3, clk3, nodes)
    clk3.t += 600.0
    ans3[A] = ans3[B] = ans3[C] = 1
    h3.probe_once()
    check(h3.restart_decision(None, 5.0) is None, "dark 10 minutes: no restart")
    h4, clk4, _l4, ans4 = tracker(nodes, return_restart_s=0.0)
    _all_down(h4, clk4, nodes)
    clk4.t += 5000.0
    ans4.update({n: 1 for n in nodes})
    h4.probe_once()
    check(h4.restart_decision(None, 5.0) is None, "--fleet-return-restart-s 0: never")
    # 10-26 19:31: cx44's REST server answered one probe after 3 h dark, stopped 10 s
    # later, and the armed restart still fired at 19:34 with the whole fleet dark
    h6, clk6, l6, ans6 = tracker(nodes)
    _all_down(h6, clk6, nodes)
    clk6.t += 1300.0
    ans6[A] = 1
    h6.probe_once()
    check(h6.return_t is not None, "one node back after 1300 s dark: armed")
    clk6.t += 2.0
    del ans6[A]
    _all_down(h6, clk6, [A])
    check(
        any(m.startswith("FLEET DARK again") for m in l6),
        "it stops answering: FLEET DARK again",
    )
    clk6.t += 200.0
    check(
        h6.restart_decision(None, 5.0) is None,
        "dark again: the pending restart is cancelled (past return_settle_s)",
    )
    clk6.t += 100.0
    ans6[A] = ans6[B] = 1
    h6.probe_once()
    check(
        h6.restart_decision(None, 5.0) == "exit",
        "the real return re-arms at once: the dark period ran on from its start",
    )
    h5, clk5, _l5, ans5 = tracker(nodes)
    _all_down(h5, clk5, [A, B, C])
    clk5.t += 5000.0
    ans5.update({A: 1, B: 1, C: 1})
    h5.probe_once()
    check(
        h5.restart_decision(None, 5.0) is None,
        "three of four down for long: no restart",
    )


def test_enable_once():
    print("one tracker per process")
    saved = nodehealth.TRACKER
    nodehealth.TRACKER = None
    real_start = NodeHealth.start
    NodeHealth.start = lambda self: None
    try:
        t1 = nodehealth.enable(["http://cx19:12048"])
        t2 = nodehealth.enable(["http://cx27:12048/gnss0_inject"], down_after_s=99.0)
        check(t1 is t2 and t1.down_after_s == 10.0, "a later chain adds its bases only")
        check(set(t1._n) == {"cx19:12048", "cx27:12048"}, "keyed by origin")
    finally:
        NodeHealth.start = real_start
        nodehealth.TRACKER = saved
        transport.set_node_hooks(None, None)


def main():
    test_classify()
    test_down_and_gate()
    test_transport_gate()
    test_probe_up()
    test_fleet_return()
    test_up_kicks_the_boresight()
    test_enable_once()
    if _fails:
        raise SystemExit("FAILED: %d" % len(_fails))
    print("ok")


if __name__ == "__main__":
    main()
