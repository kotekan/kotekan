#!/usr/bin/env python3
"""Gate: GnssFleetTrim's /set_policy must be per-chain, a silent chain's policy must expire, and
a GPU half's set_trim target must carry, in one request per round, exactly what each chain's own
target is sent.

Every broker chain thread POSTs a payload naming only its own chain:

    {"chains": {"gal_e5a": {"armed": [...], "targets": [...], ...}}}

If the stage replaced its stored policy wholesale on each POST, one armed chain would look
fine, but two would clobber each other at the policy cadence and both fall to a duty cycle set
by whoever posted last.

The axis legs 1-3 vary is the number of chains posting: leg 1 posts two and checks both
survive (a single-chain check cannot fail against either implementation); leg 2 re-posts one
and checks the other is untouched and the re-posted one is replaced, not merged. Leg 3 checks
the other direction: a chain that stops posting must leave the stored policy after
policy_ttl_s while a chain that keeps posting survives, and policy_expired must show that the
expiry fired. The expired chain must also be disarmed in FleetDll (`policy`, FleetDll's armed
set, empty) and its post targets dropped, or its last armed set would keep being integrated
and posted. Leg 4 checks the target forms: a GPU half's target {"url", "chain"} is taken beside
a tracker's own url, while one naming another chain, a malformed one, or a url named in both
forms is refused with 400 and changes nothing.

Legs 5-8 put a node in the loop: broker_fakefleet.py with chains bds_b2a, gal_e5a and gps_l5 on
two GPU halves whose inject host serves them by chain (--hosted inject), and telemetry frames
(fleetdll_gate.build_frame) fed to the gather one chain at a time. After each chain's frames the
gate waits for the gather's posts, so every request falls in a known window (a round), and the
requests are named alike in either form by node_api.canonical_requests.

  5  batched == legacy. On half 0 every chain names both its row's own set_trim and the half's
     ({url, chain}); on half 1, gal_e5a and gps_l5 name the half and bds_b2a its row. Each round
     a half's requests must carry exactly the chains that name it, each with the body that
     chain's own target was sent that round, and from the third window on in one request.
  6  replaced mid-round. gal_e5a's policy drops half 0 while a round is open there (gps_l5 has
     closed the window, the others not) and re-adds it six windows later: the open round still
     posts, without it, and it returns with the first trims formed after it was named again.
  7  refusal. gal_e6, which the node does not serve, names half 0. The node refuses the batch;
     the other chains' trims must still arrive that round and every later one, gal_e6 must be
     named in get_stats refused_chains and counted in post_refused, and no chain may go longer
     than the trackers' trim_ttl_s without trims.
  8  expired mid-feed. bds_b2a stops posting policy while its frames keep flowing; once
     policy_ttl_s passes it must leave every request, its rows' and the halves', while the other
     chains keep every round.

  9  a refusal leaves with its chain. Once every policy of legs 5-8 has expired, with nothing
     posting since, get_stats must name no refused chain and hold no post target.
 10  spread lags. gps_l5, gal_e5a and bds_b2a close their windows 0, 0.5 and 1.3 windows apart,
     in real time, and name half 1. The half must still take about one request per window, each
     chain's trims in most of them. gps_l2c names half 1 too: its senders stopped, far ahead of
     the others' windows, after its last trims formed and before the half was named, so its
     trims must never reach the half. gps_l5 also names a half the node does not have, and the
     gather must log that url's failure.

Leg 11 is a stress for the locking, meant for a TSAN build: threads post each chain's policy in
either target form and read get_stats while frames flow unpaced. The gather must keep posting
and stay up.

Runs the real binary against a real config on offset ports: `policy_armed_requested` is read
straight off the stored policy, so legs 1-4 are legible with the buffer empty. Takes about a
minute. Run it under flock /var/tmp/kvand-prs/kotekan-ports.lock where other gates share the
host.

    scripts/gnss/fleettrim_multichain_gate.py [--bin PATH] [--config PATH] [--stage NAME]
                                              [--stress-s S] [--lag-s S] [--keep DIR]
"""
import argparse
import json
import os
import random
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.dirname(os.path.dirname(HERE))
sys.path.append(os.path.join(K, "python", "scripts", "gnss"))
import kotekan_src  # noqa: E402
import fleetdll_gate  # noqa: E402
from gnss_broker import node_api  # noqa: E402

REST, RECV, SERVE, FAKE = 12451, 11460, 11461, 12452
TTL = 6.0
TRIM_TTL_S = 4.0  # the trackers' trim_ttl_s (gnssTrackChain.hpp)
NODE = "http://127.0.0.1:%d" % FAKE
SERVED = ["bds_b2a", "gal_e5a", "gps_l5"]  # the node's chains, on both halves
REFUSED = "gal_e6"  # a chain the node does not serve
DEAD = "gps_l2c"  # a chain the node serves whose senders stop before leg 10
LAGS = {"gps_l5": 0.0, "gal_e5a": 0.5, "bds_b2a": 1.3}  # leg 10, in windows
# One window, 4 records x 2048 hops x 5.12 us = 41.94 ms.
WIN_S = fleetdll_gate.N_REC * fleetdll_gate.HOPS_PER_RECORD * 5.12e-6
INSTANCES = [("node0.0", 7), ("node1.0", 6)]  # two senders per chain: min_instances
PRN_ROWS = [4, 9, 27, 3, 16, 5, 12, 18, 21, 24, 30] + [0] * 5
ARMED = [3, 4, 9, 27]
# bufferSend's wire header without the config tracker, and the GnssChanMetadata it carries.
SEND_HDR = struct.Struct("<II")
META = struct.Struct("<qI")
_KEY = re.compile(r"/gnss(\d+)/inject/([a-z0-9_]+)/set_trim$")


def _get(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url, payload, timeout=5.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _post_status(url, payload):
    try:
        return _post(url, payload)
    except urllib.error.HTTPError as e:
        return e.code


def policy_for(chain, armed, targets=("http://127.0.0.1:9/set_trim",), p_floor_abs=0.0):
    return {
        "chains": {
            chain: {
                "armed": list(armed),
                "gain_per_s": 2.5,
                "leak_per_s": 0.5,
                "clamp": 3.0,
                "spacing": 0.5,
                "p_floor_abs": p_floor_abs,
                "targets": list(targets),
            }
        }
    }


def row_target(half, chain):
    return "%s/gnss%d_%s_inject/set_trim" % (NODE, half, chain)


def half_target(half, chain):
    return {"url": "%s/gnss%d_inject/set_trim" % (NODE, half), "chain": chain}


class Feed(object):
    """The gather's telemetry listener, fed one chain's frames for one window at a time."""

    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.rng = random.Random(20261119)
        self.sent = {}

    def send(self, chain, win):
        for i, (inst, n_chan) in enumerate(INSTANCES):
            fr = fleetdll_gate.build_frame(
                chain,
                inst,
                i,
                n_chan,
                win,
                win * 10 + i,
                PRN_ROWS,
                0xF,
                self.rng,
                {"energy": set()},
            )
            wstart0 = win * fleetdll_gate.N_REC * fleetdll_gate.HOPS_PER_RECORD
            wstart0 *= fleetdll_gate.FFT_LEN
            self.sock.sendall(
                SEND_HDR.pack(META.size, len(fr)) + META.pack(wstart0, 0) + fr
            )
            self.sent[chain] = self.sent.get(chain, 0) + 1


class NodeLog(object):
    """The fake node's request log (broker_fakefleet.py --log), read as it grows."""

    def __init__(self, path):
        self.path = path
        self.off = 0
        self.lines = []

    def poll(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            f.seek(self.off)
            data = f.read()
        n = data.rfind("\n") + 1
        self.off += len(data[:n].encode())
        self.lines += [json.loads(l) for l in data[:n].splitlines()]


def trims_of(lines):
    """[(half, chain, body, status, form, n, time)] of the set_trim requests in `lines`, one per
    chain: a half's request as each chain's own (node_api.canonical_requests), n the chains it
    carried. form is "row" for a row's own path, "half" for its half's."""
    out = []
    for l in lines:
        if l.get("m") != "POST" or not l["p"].endswith("/set_trim"):
            continue
        form = (
            "row"
            if re.match(r"^/gnss\d+_[a-z0-9]+_[a-z0-9]+_inject/", l["p"])
            else "half"
        )
        reqs = node_api.canonical_requests(NODE + l["p"], l["b"])
        for key, body in reqs:
            m = _KEY.search(key)
            h, c = (int(m.group(1)), m.group(2)) if m else (None, key)
            out.append((h, c, body, l["s"], form, len(reqs), l.get("t", 0.0)))
    return out


def _answers(url):
    try:
        _get(url, timeout=1.0)
        return True
    except Exception:
        return False


def wait_for(cond, timeout, step=0.002):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if cond():
            return True
        time.sleep(step)
    return cond()


def data_legs(base, log, fails, pace_s):
    """Legs 5-8 in one feed of windows. A chain's own targets (its rows' set_trim) define its
    rounds: each window, every half that the chain names when its window closes must take, in
    that round, exactly the body its rows were sent."""
    targets = {
        "gps_l5": [
            row_target(0, "gps_l5"),
            half_target(0, "gps_l5"),
            half_target(1, "gps_l5"),
        ],
        "gal_e5a": [
            row_target(0, "gal_e5a"),
            half_target(0, "gal_e5a"),
            half_target(1, "gal_e5a"),
        ],
        "bds_b2a": [
            row_target(0, "bds_b2a"),
            half_target(0, "bds_b2a"),
            row_target(1, "bds_b2a"),
        ],
    }
    # chain -> the time of its last policy post, for each chain that keeps posting
    posting = {"gps_l5": 0.0, "gal_e5a": 0.0, "bds_b2a": 0.0}
    order = ["gps_l5", "gal_e5a", "bds_b2a", REFUSED]

    def post_policy(c):
        code = _post_status(
            base + "/set_policy", policy_for(c, ARMED, targets[c], p_floor_abs=1e-9)
        )
        if code != 200:
            fails.append("LEGS 5-8: set_policy for %s answered %s" % (c, code))
        posting[c] = time.time()

    for c in sorted(posting):
        post_policy(c)
    feed = Feed(RECV)
    stopped = None  # when bds_b2a stopped posting policy (leg 8)
    gone = None  # the first window after that with no bds_b2a round
    # (window, the set_trim requests, {chain: halves it named at its close}, near expiry)
    rounds = []
    late = 0  # windows whose halves' requests never came
    w = 0
    while gone is None or w < gone + 8:
        if w > 400:
            fails.append(
                "LEG 8: bds_b2a's rounds never stopped (policy_ttl_s %.0f s)" % TTL
            )
            break
        if w == 30:
            # Leg 7: a chain the node does not serve names half 0. Leg 8: bds_b2a goes silent.
            targets[REFUSED] = [half_target(0, REFUSED)]
            posting[REFUSED] = 0.0
            del posting["bds_b2a"]
            stopped = time.time()
        for c in sorted(posting):
            if time.time() - posting[c] > 1.0:
                post_policy(c)
        start = len(log.lines)
        named = {}
        # Near its expiry, bds_b2a may have its last round or not, and its policy may expire
        # between its window's close and its halves' rounds, which then go without it.
        near = stopped is not None and time.time() - stopped > TTL - 2.0
        for c in order:
            feed.send(c, w)
            if not wait_for(
                lambda: (_get(base + "/get_stats")["chains"].get(c) or {}).get(
                    "frames", 0
                )
                >= feed.sent[c],
                10.0,
            ):
                fails.append(
                    "LEGS 5-8: the gather did not fold %s's window %d" % (c, w)
                )
                return w + 1
            # Leg 6: gal_e5a leaves half 0 while gps_l5 has closed this window and the other
            # chains have not, and comes back six windows later.
            if c == "gps_l5" and w in (20, 26):
                targets["gal_e5a"] = [
                    row_target(0, "gal_e5a"),
                    half_target(1, "gal_e5a"),
                ]
                if w == 26:
                    targets["gal_e5a"].insert(1, half_target(0, "gal_e5a"))
                post_policy("gal_e5a")
            named[c] = set(
                int(re.search(r"/gnss(\d+)_inject/", t["url"]).group(1))
                for t in targets.get(c, [])
                if isinstance(t, dict)
            )
            if c == REFUSED:  # no own target: never taken (leg 7)
                continue
            dll = _get(base + "/get_dll?chain=" + c).get(c) or {}
            if not any(isinstance(v, dict) and "trim_chips" in v for v in dll.values()):
                continue
            rows = len([t for t in targets[c] if isinstance(t, str)])

            def rows_in():
                log.poll()
                got = trims_of(log.lines[start:])
                return len([x for x in got if x[4] == "row" and x[1] == c]) >= rows

            if not wait_for(rows_in, 0.3 if near and c == "bds_b2a" else 5.0) and not (
                near and c == "bds_b2a"
            ):
                fails.append(
                    "LEG 5: window %d: %s's own targets were not posted" % (w, c)
                )
            time.sleep(pace_s)

        def halves_in():
            log.poll()
            got = trims_of(log.lines[start:])
            own = set(x[1] for x in got if x[4] == "row")
            took = set((x[0], x[1]) for x in got if x[4] == "half" and x[3] == 200)
            return all(
                (h, c) in took
                for c in own
                for h in named.get(c, ())
                if not (near and c == "bds_b2a")
            )

        if not wait_for(halves_in, 5.0):
            late += 1
            if late > 3:  # judged below; another 5 s wait per window adds nothing
                fails.append("LEGS 5-8: the feed stopped at window %d" % w)
                break
        time.sleep(max(0.05, pace_s))  # a refused chain's lone post follows the batch
        log.poll()
        rec = trims_of(log.lines[start:])
        rounds.append((w, rec, named, near))
        if stopped is not None and gone is None and time.time() - stopped > TTL:
            if not any(x[1] == "bds_b2a" for x in rec):
                gone = w
        w += 1

    first_refusal = None
    n_one = n_rounds = 0
    arrivals = {}  # (half, chain, form) -> times the node took its trims
    for win, rec, named, near in rounds:
        own = {}  # chain -> the body its rows were sent this round
        took = {}  # (half, chain) -> the body the half took
        for h, c, body, st, form, n, t in rec:
            if st == 200:
                arrivals.setdefault((h, c, form), []).append(t)
            if form == "row":
                if c in own and own[c] != body:
                    fails.append(
                        "LEG 5: window %d: %s's rows were sent different bodies"
                        % (win, c)
                    )
                own[c] = body
            elif st == 200:
                if (h, c) in took:
                    fails.append(
                        "LEG 5: window %d: half %d took %s twice" % (win, h, c)
                    )
                took[(h, c)] = body
        want = set((h, c) for c in own for h in named.get(c, ()))
        if near:
            want -= set(k for k in want - set(took) if k[1] == "bds_b2a")
        if set(took) != want:
            fails.append(
                "LEGS 5-8: window %d: the halves took %s; the chains with a round that name "
                "them are %s" % (win, sorted(took), sorted(want))
            )
        for (h, c), body in sorted(took.items()):
            if c in own and body != own[c]:
                fails.append(
                    "LEG 5: window %d: half %d's request carried %s %s, its rows %s"
                    % (win, h, c, json.dumps(body)[:100], json.dumps(own[c])[:100])
                )
        if first_refusal is None and any(x[4] == "half" and x[3] == 400 for x in rec):
            first_refusal = win
        # One request per round, from the third window and outside the refusal's first round:
        # a request carrying n chains gives n entries, each with n.
        for h in (0, 1):
            ok = [x for x in rec if x[0] == h and x[4] == "half" and x[3] == 200]
            if win < 2 or win == first_refusal or not ok:
                continue
            n_rounds += 1
            reqs = sum(1.0 / x[5] for x in ok)
            if abs(reqs - 1.0) < 1e-9:
                n_one += 1
            else:
                fails.append(
                    "LEG 5: window %d: half %d took %s in %.0f requests, not one"
                    % (win, h, sorted(x[1] for x in ok), reqs)
                )
    print(
        "LEGS 5-8 %d windows; %d half rounds after warm-up, %d in one request; first refusal "
        "in window %s; bds_b2a's rounds stopped by window %s"
        % (len(rounds), n_rounds, n_one, first_refusal, gone)
    )
    if n_rounds < 2 * (len(rounds) - 4):
        fails.append("LEG 5: %d half rounds over %d windows" % (n_rounds, len(rounds)))
    # Leg 6: the open round posted without gal_e5a, which was back six windows later.
    by_win = dict(
        (win, set((x[0], x[1]) for x in rec if x[4] == "half" and x[3] == 200))
        for win, rec, _, _ in rounds
    )
    if not (
        {(0, "gps_l5"), (0, "bds_b2a")} <= by_win.get(20, set())
        and (0, "gal_e5a") not in by_win.get(20, set()) | by_win.get(25, set())
        and (0, "gal_e5a") in by_win.get(26, set())
    ):
        fails.append(
            "LEG 6: half 0 took %s in window 20, %s in 25 and %s in 26"
            % (
                sorted(by_win.get(20, ())),
                sorted(by_win.get(25, ())),
                sorted(by_win.get(26, ())),
            )
        )
    # Leg 7: the refusal.
    st = _get(base + "/get_stats")
    u0 = half_target(0, REFUSED)["url"]
    print(
        "LEG 7 refused_chains %s, post_refused %s, post_halves %s carrying %s chains"
        % (
            st.get("refused_chains"),
            st.get("post_refused"),
            st.get("post_halves"),
            st.get("post_half_chains"),
        )
    )
    if first_refusal is None:
        fails.append("LEG 7: the node never refused a request from the gather")
    if REFUSED not in (st.get("refused_chains") or {}).get(u0, []):
        fails.append("LEG 7: refused_chains does not name %s at %s" % (REFUSED, u0))
    if not st.get("post_refused"):
        fails.append("LEG 7: post_refused is %s" % st.get("post_refused"))
    for (h, c, form), ts in sorted(arrivals.items()):
        if c == REFUSED:
            fails.append("LEG 7: the node took %s's trims" % REFUSED)
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        if c != "bds_b2a" and gaps and max(gaps) >= TRIM_TTL_S:
            fails.append(
                "LEG 7: half %d went %.1f s without %s's trims (%s target; trim_ttl_s %.0f)"
                % (h, max(gaps), c, form, TRIM_TTL_S)
            )
    # Leg 8: bds_b2a in no request once its rounds stop, and the stage says it expired.
    if gone is None:
        fails.append("LEG 8: bds_b2a never expired")
        return w
    for win, rec, _, _ in rounds:
        if win > gone and any(x[1] == "bds_b2a" for x in rec):
            fails.append(
                "LEG 8: window %d: bds_b2a's trims were posted after it expired" % win
            )
    if "bds_b2a" in st.get("policy_armed_requested", {}):
        fails.append(
            "LEG 8: bds_b2a's rounds stopped but its policy stands: %s"
            % st.get("policy_armed_requested")
        )
    return w


def fold(base, feed, c, w):
    """Send chain c's window w and wait until the gather has folded it."""

    def frames():
        return (_get(base + "/get_stats")["chains"].get(c) or {}).get("frames", 0)

    want = frames() + len(INSTANCES)
    feed.send(c, w)
    return wait_for(lambda: frames() >= want, 10.0)


def expired_refusal(base, fails, w):
    """Leg 9: every policy expires, a frame runs the sweep, and no refusal or target is left."""
    time.sleep(TTL + 1.5)
    if not fold(base, Feed(RECV), "gps_l5", w):
        fails.append("LEG 9: the gather did not fold gps_l5's window %d" % w)
    st = _get(base + "/get_stats")
    print(
        "LEG 9 every policy expired: policy_armed_requested %s, post_targets %s, "
        "refused_chains %s"
        % (
            st.get("policy_armed_requested"),
            st.get("post_targets"),
            st.get("refused_chains"),
        )
    )
    if st.get("policy_armed_requested") or st.get("post_targets"):
        fails.append(
            "LEG 9: policies %s and %s post target(s) outlived policy_ttl_s"
            % (st.get("policy_armed_requested"), st.get("post_targets"))
        )
    if st.get("refused_chains"):
        fails.append(
            "LEG 9: refused_chains %s names chains whose policies expired"
            % st.get("refused_chains")
        )


def spread_lags(base, log, gather_log, fails, w0, seconds):
    """Leg 10: chains closing at spread lags, and a silent chain far ahead, on half 1."""
    feed = Feed(RECV)
    _post(base + "/set_policy", policy_for(DEAD, ARMED, [], p_floor_abs=1e-9))
    w = w0 + 5000
    for _ in range(80):
        if not fold(base, feed, DEAD, w):
            fails.append("LEG 10: the gather did not fold %s's window %d" % (DEAD, w))
            return
        w += 1
        dll = _get(base + "/get_dll?chain=" + DEAD).get(DEAD) or {}
        if any(isinstance(v, dict) and "trim_chips" in v for v in dll.values()):
            break
    else:
        fails.append("LEG 10: %s formed no trims" % DEAD)
        return
    fold(base, feed, DEAD, w)  # one more close with trims, then its senders stop

    absent = half_target(2, "gps_l5")

    def post_all():
        for c in sorted(LAGS) + [DEAD]:
            t = [half_target(1, c)] + ([absent] if c == "gps_l5" else [])
            _post(base + "/set_policy", policy_for(c, ARMED, t, p_floor_abs=1e-9))

    log.poll()
    start = len(log.lines)
    post_all()
    n = int(seconds / WIN_S) + 2
    t0 = time.time() + 0.2
    events = sorted(
        (t0 + (k + lag) * WIN_S, c, w0 + k) for c, lag in LAGS.items() for k in range(n)
    )
    posted = time.time()
    for t, c, win in events:
        dt = t - time.time()
        if dt > 0:
            time.sleep(dt)
        feed.send(c, win)
        if time.time() - posted > 1.5:
            post_all()
            posted = time.time()
    time.sleep(0.5)
    log.poll()
    rec = [x for x in trims_of(log.lines[start:]) if x[4] == "half" and x[0] == 1]
    ta, tb = t0 + 1.5, t0 + n * WIN_S
    ok = [x for x in rec if x[3] == 200 and ta <= x[6] <= tb]
    wins = (tb - ta) / WIN_S
    reqs = sum(1.0 / x[5] for x in ok)
    took = dict((c, sorted(x[6] for x in ok if x[1] == c)) for c in LAGS)
    print(
        "LEG 10 %.0f windows: half 1 took %.1f requests per window; per window %s"
        % (
            wins,
            reqs / wins,
            ", ".join(
                "%s %.2f" % (c, len(ts) / wins) for c, ts in sorted(took.items())
            ),
        )
    )
    if not 0.7 <= reqs / wins <= 1.3:
        fails.append(
            "LEG 10: half 1 took %.2f requests per window, not about one"
            % (reqs / wins)
        )
    for c, ts in sorted(took.items()):
        gaps = [b - a for a, b in zip([ta] + ts, ts + [tb])]
        if len(ts) < 0.7 * wins or max(gaps) >= TRIM_TTL_S:
            fails.append(
                "LEG 10: half 1 took %s's trims %d times in %.0f windows, longest gap %.2f s"
                % (c, len(ts), wins, max(gaps))
            )
    if any(x[1] == DEAD for x in rec):
        fails.append(
            "LEG 10: half 1 was sent %s's trims, formed before its policy named the half"
            % DEAD
        )
    with open(gather_log) as f:
        if not any("%s failed" % absent["url"] in line for line in f):
            fails.append("LEG 10: the gather did not log %s's failure" % absent["url"])


def stress(base, seconds, fails):
    """Leg 11: policies in either form and get_stats from several threads while frames flow."""
    stop = threading.Event()
    errors = []

    def poster(c, k):
        rng = random.Random(k)
        while not stop.is_set():
            t = [row_target(0, c)] if rng.random() < 0.3 else []
            t += [half_target(h, c) for h in (0, 1) if rng.random() < 0.7]
            try:
                _post(base + "/set_policy", policy_for(c, ARMED, t, p_floor_abs=1e-9))
            except Exception as e:  # a refusal here is a failure: every payload is valid
                errors.append("%s: %s" % (c, e))
            time.sleep(rng.random() * 0.02)

    def reader():
        while not stop.is_set():
            try:
                _get(base + "/get_stats")
                _get(base + "/get_dll")
            except Exception as e:
                errors.append("get: %s" % e)

    before = _get(base + "/get_stats").get("post_requests", 0)
    ts = [
        threading.Thread(target=poster, args=(c, k))
        for k, c in enumerate(SERVED + [REFUSED])
    ]
    ts.append(threading.Thread(target=reader))
    for t in ts:
        t.start()
    feed = Feed(RECV)
    w, t0 = 1000, time.time()
    while time.time() - t0 < seconds:
        for c in SERVED + [REFUSED]:
            feed.send(c, w)
        w += 1
        time.sleep(0.042)
    stop.set()
    for t in ts:
        t.join()
    after = _get(base + "/get_stats").get("post_requests", 0)
    print(
        "LEG 11 %d windows in %.0f s, %d posts, %d errors"
        % (w - 1000, seconds, after - before, len(errors))
    )
    if errors:
        fails.append("LEG 11: %d request(s) failed, e.g. %s" % (len(errors), errors[0]))
    if after <= before:
        fails.append("LEG 11: the gather posted nothing under load")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bin",
        help="the binary (default: build_nodpdk/kotekan/kotekan in the kotekan tree)",
    )
    ap.add_argument(
        "--config", default=os.path.join(K, "config/generated/chord_gnss_gather.yaml")
    )
    ap.add_argument("--stage", default="fleet_trim")
    ap.add_argument(
        "--stress-s", type=float, default=3.0, help="leg 11's length, 0 = skip"
    )
    ap.add_argument("--lag-s", type=float, default=6.0, help="leg 10's length")
    ap.add_argument(
        "--pace-s",
        type=float,
        default=0.005,
        help="pause after each chain's posts (legs 5-8)",
    )
    ap.add_argument(
        "--keep", help="keep the gather's and the node's logs in this directory"
    )
    a = ap.parse_args()
    if a.bin is None:
        a.bin = kotekan_src.src("build_nodpdk", "kotekan", "kotekan")

    import yaml

    work = tempfile.mkdtemp(prefix="fleettrim_gate_")
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    cfg["rest_server"]["port"] = REST
    cfg["telem_gather"]["serve_port"] = SERVE
    cfg["telem_recv"]["listen_port"] = RECV
    cfg[a.stage]["policy_ttl_s"] = TTL
    cfg[a.stage]["trim_state_file"] = os.path.join(work, "trims.json")
    cfg_path = os.path.join(work, "gather.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    log = open(os.path.join(work, "gather.log"), "w")
    node_log = os.path.join(work, "node.jsonl")
    node_out = open(os.path.join(work, "node.out"), "w")
    fake = subprocess.Popen(
        [
            sys.executable,
            os.path.join(HERE, "broker_fakefleet.py"),
            "--port",
            str(FAKE),
            "--chains",
            ",".join(SERVED + [DEAD]),
            "--halves",
            "2",
            "--hosted",
            "inject",
            "--log",
            node_log,
        ],
        stdout=node_out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc = subprocess.Popen(
        [a.bin, "--config", cfg_path, "--bind-address", "0.0.0.0:%d" % REST],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base = "http://127.0.0.1:%d/%s" % (REST, a.stage)
    fails = []
    try:
        for _ in range(100):
            try:
                _get(base + "/get_stats", timeout=1.0)
                break
            except Exception:
                time.sleep(0.2)
        else:
            print(
                "FAILED: gather did not serve REST on :%d -- see %s" % (REST, log.name)
            )
            return 1

        # ── Leg 1: two chains, posted separately, must both stay armed ────────────────
        _post(base + "/set_policy", policy_for("gps_l5", [1, 3, 10]))
        _post(base + "/set_policy", policy_for("gal_e5a", [6, 12, 19, 29]))
        st = _get(base + "/get_stats")
        req = st.get("policy_armed_requested", {})
        print("LEG 1 two chains posted separately -> policy_armed_requested = %s" % req)
        if req.get("gps_l5") != 3 or req.get("gal_e5a") != 4:
            fails.append(
                "LEG 1: expected gps_l5=3 and gal_e5a=4 armed, got %s. The second "
                "chain's POST wiped the first -- the wholesale-replace clobber." % req
            )
        if st.get("post_targets", 0) != 2:
            fails.append(
                "LEG 1: expected 2 post targets (one per chain), got %s -- the target "
                "list is being replaced wholesale even if the policy is not."
                % st.get("post_targets")
            )

        # ── Leg 2: re-posting one chain must not disturb the other, and must still
        #    disarm a PRN that chain stopped naming (per-chain replace, not merge) ─────
        _post(base + "/set_policy", policy_for("gps_l5", [1]))
        st = _get(base + "/get_stats")
        req = st.get("policy_armed_requested", {})
        print(
            "LEG 2 gps_l5 re-posted with 1 PRN      -> policy_armed_requested = %s"
            % req
        )
        if req.get("gal_e5a") != 4:
            fails.append(
                "LEG 2: gal_e5a should be untouched at 4, got %s" % req.get("gal_e5a")
            )
        if req.get("gps_l5") != 1:
            fails.append(
                "LEG 2: gps_l5 should REPLACE to 1 (not merge to 3), got %s -- a merge "
                "leaves a PRN armed forever after policy stops naming it."
                % req.get("gps_l5")
            )

        # ── Leg 3: a chain that stops posting must expire. gps_l5 keeps posting across the
        #    TTL, which shows the sweep is selective, not a global timeout; with no frames
        #    flowing, its POSTs are also what run the sweep (rearm()). ──────────────────
        t0 = time.time()
        while time.time() - t0 < TTL + 4.0:
            _post(base + "/set_policy", policy_for("gps_l5", [1]))
            time.sleep(1.0)
        st = _get(base + "/get_stats")
        req = st.get("policy_armed_requested", {})
        print(
            "LEG 3 gal_e5a silent %.0fs (gps_l5 alive) -> policy_armed_requested = %s, "
            "policy_expired = %s" % (TTL + 4.0, req, st.get("policy_expired"))
        )
        if "gal_e5a" in req:
            fails.append(
                "LEG 3: gal_e5a POSTed nothing for >%.0fs and is still armed (%s). A "
                "dead broker chain thread would command forever." % (TTL, req)
            )
        if req.get("gps_l5") != 1:
            fails.append(
                "LEG 3: gps_l5 kept posting and must survive, got %s -- the expiry "
                "swept a live chain." % req.get("gps_l5")
            )
        dll_armed = st.get("policy", {}).get("gal_e5a", {}).get("armed")
        if dll_armed != []:
            fails.append(
                "LEG 3: gal_e5a expired but FleetDll still arms %s -- its trims keep "
                "integrating on the last policy." % dll_armed
            )
        if st.get("post_targets", 0) != 1:
            fails.append(
                "LEG 3: expected 1 post target (gps_l5's) after gal_e5a expired, got %s -- "
                "the silent chain's trims are still posted." % st.get("post_targets")
            )
        if not st.get("policy_expired"):
            fails.append(
                "LEG 3: policy_expired is %s; the expiry never fired, so this leg "
                "could not have failed." % st.get("policy_expired")
            )

        # ── Leg 4: target forms ──────────────────────────────────────────────────────────
        good = [
            row_target(0, "gps_l5"),
            half_target(0, "gps_l5"),
            half_target(1, "gps_l5"),
        ]
        code = _post_status(base + "/set_policy", policy_for("gps_l5", [1], good))
        st = _get(base + "/get_stats")
        print(
            "LEG 4 a row and two halves -> %s, post_targets %s"
            % (code, st.get("post_targets"))
        )
        if code != 200 or st.get("post_targets") != 3:
            fails.append(
                "LEG 4: {url, chain} targets beside a url: %s, post_targets %s"
                % (code, st.get("post_targets"))
            )
        bad = {
            "another chain": [{"url": half_target(0, "x")["url"], "chain": "gal_e5a"}],
            "no chain": [{"url": half_target(0, "x")["url"]}],
            "an extra key": [dict(half_target(0, "gps_l5"), extra=1)],
            "a url that is not a string": [{"url": 5, "chain": "gps_l5"}],
            "a number": [7],
        }
        for what, t in sorted(bad.items()):
            code = _post_status(base + "/set_policy", policy_for("gps_l5", [1, 3], t))
            if code != 400:
                fails.append(
                    "LEG 4: a target with %s was answered %s, not 400" % (what, code)
                )
        both = policy_for("gal_e5a", [6], [half_target(0, "gps_l5")["url"]])
        code = _post_status(base + "/set_policy", both)
        if code != 400:
            fails.append(
                "LEG 4: a url named both as a url and as a half's was answered %s"
                % code
            )
        st = _get(base + "/get_stats")
        if (
            st.get("post_targets") != 3
            or st.get("policy_armed_requested", {}).get("gps_l5") != 1
        ):
            fails.append(
                "LEG 4: a refused policy changed the stage: post_targets %s, armed %s"
                % (st.get("post_targets"), st.get("policy_armed_requested"))
            )

        # ── Legs 5-8: the node in the loop ───────────────────────────────────────────────
        if not wait_for(lambda: _answers(NODE + "/gnss0_inject/chains"), 10.0, 0.2):
            print(
                "FAILED: the fake node does not answer on :%d -- see %s"
                % (FAKE, node_out.name)
            )
            return 1
        nlog = NodeLog(node_log)
        w = data_legs(base, nlog, fails, a.pace_s)
        expired_refusal(base, fails, w)
        spread_lags(base, nlog, log.name, fails, w + 1, a.lag_s)
        if a.stress_s > 0:
            stress(base, a.stress_s, fails)
        if proc.poll() is not None:
            fails.append("the gather exited with %s" % proc.returncode)
    finally:
        for p in (proc, fake):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
            p.wait(timeout=10)
        log.close()
        node_out.close()
        if a.keep:
            os.makedirs(a.keep, exist_ok=True)
            for f in os.listdir(work):
                shutil.copy(os.path.join(work, f), a.keep)
        shutil.rmtree(work, ignore_errors=True)

    print()
    for f in fails:
        print("  FAIL " + f)
    print(
        "fleettrim_multichain_gate: %s"
        % ("FAILED (%d)" % len(fails) if fails else "PASS")
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
