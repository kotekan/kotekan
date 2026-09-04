#!/usr/bin/env python3
"""Gate: /set_policy must be per-chain, and a silent chain must disarm (#49 arming).

WHY THIS EXISTS. Every broker chain thread POSTs a payload naming ONLY its own chain:

    {"chains": {"gal_e5a": {"armed": [...], "targets": [...], ...}}}

while GnssFleetTrim used to store it with `_policy = got` -- a wholesale replace. With ONE
armed chain that is invisible, and gps_l5 was the only armed chain for a day. The moment a
second chain is armed the two clobber each other at the policy cadence and both fall to a
duty cycle set by whoever posted last. It would have presented as "arming gal_e5a broke
gps_l5", which is the most expensive possible way to learn it.

⚠️ WHAT A GATE MUST VARY. The axis here is the NUMBER OF CHAINS POSTING, so this posts two
and checks BOTH survive. A single-chain check cannot fail against either implementation and
would have passed happily all day. Leg 3 exists for the same reason in reverse: it proves
the anti-latch property the wholesale replace used to buy is still bought, by STOPPING a
chain and watching it go -- an expiry that never fires is a gate that cannot fail.

Runs the real binary against a real config on offset ports; no data flows, which is the
point (`policy_armed_requested` is read straight off _policy, so it is legible with the
buffer empty). ~25 s.

    scripts/gnss/fleettrim_multichain_gate.py [--bin PATH] [--config PATH]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REST, RECV, SERVE = 12451, 11460, 11461
TTL = 6.0


def _get(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url, payload, timeout=5.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def policy_for(chain, armed, targets=("http://127.0.0.1:9/set_trim",)):
    return {"chains": {chain: {
        "armed": list(armed), "gain_per_s": 2.5, "leak_per_s": 0.5,
        "clamp": 3.0, "spacing": 0.5, "targets": list(targets)}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(K, "build_nodpdk/kotekan/kotekan"))
    ap.add_argument("--config", default=os.path.join(K, "config/generated/chord_gnss_gather.yaml"))
    ap.add_argument("--stage", default="fleet_trim")
    a = ap.parse_args()

    import yaml
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    cfg["rest_server"]["port"] = REST
    cfg["telem_gather"]["serve_port"] = SERVE
    cfg["telem_recv"]["listen_port"] = RECV
    cfg[a.stage]["policy_ttl_s"] = TTL
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()

    log = open(os.path.join(tempfile.gettempdir(), "fleettrim_gate.log"), "w")
    proc = subprocess.Popen([a.bin, "--config", tmp.name, "--bind-address", "0.0.0.0:%d" % REST],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
            print("FAILED: gather did not serve REST on :%d -- see %s" % (REST, log.name))
            return 1

        # ── LEG 1: two chains, posted separately, must BOTH stay armed ────────────────
        _post(base + "/set_policy", policy_for("gps_l5", [1, 3, 10]))
        _post(base + "/set_policy", policy_for("gal_e5a", [6, 12, 19, 29]))
        st = _get(base + "/get_stats")
        req = st.get("policy_armed_requested", {})
        print("LEG 1 two chains posted separately -> policy_armed_requested = %s" % req)
        if req.get("gps_l5") != 3 or req.get("gal_e5a") != 4:
            fails.append("LEG 1: expected gps_l5=3 and gal_e5a=4 armed, got %s. The second "
                         "chain's POST wiped the first -- the wholesale-replace clobber." % req)
        if st.get("post_targets", 0) != 2:
            fails.append("LEG 1: expected 2 post targets (one per chain), got %s -- the target "
                         "list is being replaced wholesale even if the policy is not."
                         % st.get("post_targets"))

        # ── LEG 2: re-posting ONE chain must not disturb the other, and must still
        #    expire a PRN that chain stopped naming (per-chain replace, not merge) ─────
        _post(base + "/set_policy", policy_for("gps_l5", [1]))
        st = _get(base + "/get_stats")
        req = st.get("policy_armed_requested", {})
        print("LEG 2 gps_l5 re-posted with 1 PRN      -> policy_armed_requested = %s" % req)
        if req.get("gal_e5a") != 4:
            fails.append("LEG 2: gal_e5a should be untouched at 4, got %s" % req.get("gal_e5a"))
        if req.get("gps_l5") != 1:
            fails.append("LEG 2: gps_l5 should REPLACE to 1 (not merge to 3), got %s -- a merge "
                         "leaves a PRN armed forever after policy stops naming it."
                         % req.get("gps_l5"))

        # ── LEG 3: a chain that STOPS posting must disarm. Keep gps_l5 alive across the
        #    TTL so the sweep is proven selective, not a global timeout. ───────────────
        t0 = time.time()
        while time.time() - t0 < TTL + 4.0:
            _post(base + "/set_policy", policy_for("gps_l5", [1]))
            time.sleep(1.0)
        st = _get(base + "/get_stats")
        req = st.get("policy_armed_requested", {})
        print("LEG 3 gal_e5a silent %.0fs (gps_l5 alive) -> policy_armed_requested = %s, "
              "policy_expired = %s" % (TTL + 4.0, req, st.get("policy_expired")))
        if "gal_e5a" in req:
            fails.append("LEG 3: gal_e5a POSTed nothing for >%.0fs and is still armed (%s). A "
                         "dead broker chain thread would command forever." % (TTL, req))
        if req.get("gps_l5") != 1:
            fails.append("LEG 3: gps_l5 kept posting and must survive, got %s -- the expiry "
                         "swept a live chain." % req.get("gps_l5"))
        if not st.get("policy_expired"):
            fails.append("LEG 3: policy_expired is %s; the expiry never fired, so this leg "
                         "could not have failed." % st.get("policy_expired"))
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        proc.wait(timeout=10)
        log.close()
        os.unlink(tmp.name)

    print()
    for f in fails:
        print("  FAIL " + f)
    print("fleettrim_multichain_gate: %s" % ("FAILED (%d)" % len(fails) if fails else "PASS"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
