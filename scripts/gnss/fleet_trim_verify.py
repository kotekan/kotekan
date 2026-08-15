#!/usr/bin/env python3
"""POST-RESTART CHECK for the fleet code loop's last hop (task #51 F4 step 1).

Everything upstream of the trackers is measured and gated offline: the fold and the integrator
against the Python arm on identical bytes (fleetdll_gate.py), the actuator's SIGN through the
shipped despread (trim_sign_gate.py), and the post rate on sky (23.85/s/path). The one link
never exercised on hardware is the last one -- that a POST to /set_trim reaches a live
tracker's `trim[]` and is applied to what it despreads. The running fleet predated the
endpoint entirely, so this is the first thing to run after the nodes come up on the new build.

    ./fleet_trim_verify.py                      # read-only: is the endpoint there, is it sane
    ./fleet_trim_verify.py --write              # ALSO command a small trim and read it back

⚠️ --write TOUCHES A LIVE TRACKER. It commands one PRN a small trim (default 0.02 chips, well
inside the current tracking noise), reads it back, and restores zero. Two things make that
safe rather than brave:
  * `trim_ttl_s` (4 s on the fleet) ZEROES any stamped trim that stops being refreshed, so a
    script killed mid-run self-clears within a few frames. The safety property is in the
    tracker, not in this script's error handling.
  * the value is small enough that even if it were left standing it is ~0.16 of the drift the
    loop corrects in one second.

WHAT IT ASSERTS, and why each one:
  1. /set_trim EXISTS on every instance -- a 404 means the node is on the old build, and a
     silent 404 is exactly how a "deployed" loop actuates nothing.
  2. Every instance's endpoint is DISTINCT. The stage's default is one fixed path, so an
     instance whose config lacks set_trim_endpoint registers the shared default and the last
     registration wins (restServer: "already exists, overriding old callback!!"). That would
     land one chain's trim on another chain's PRN slots.
  3. `enabled` (the IN-TRACKER loop, `code_trim`) is FALSE. Two writers on one `trim[]` vector
     is a race, and the fleet loop must be the only one.
  4. With --write: the value read back equals the value posted, and `posts` incremented.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker.transport import parse_endpoints  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None


def get(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as h:
        return json.loads(h.read().decode())


def post(url, payload, timeout=5.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as h:
        return h.status


def trackers_from_chains(path, chains):
    """The SAME endpoint list the broker uses -- brace-expanded by the SAME function.

    Re-deriving it here would be a second copy of a deployment fact, which is the thing #51 F3
    deliberately avoided by having the broker publish targets to the controller.
    """
    if yaml is None:
        raise SystemExit("pyyaml not available; pass --tracker explicitly")
    cfg = yaml.safe_load(open(path))
    out = {}
    for name, ch in (cfg.get("chains") or cfg).items():
        if not isinstance(ch, dict) or "trackers" not in ch:
            continue
        if chains and name not in chains:
            continue
        out[name] = parse_endpoints(ch["trackers"], "")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains-yaml",
                    default=os.path.join(HERE, "..", "..", "config", "gnss_chains_chord.yaml"))
    ap.add_argument("--chain", action="append", default=[],
                    help="restrict to these chains (default: all in the yaml)")
    ap.add_argument("--tracker", action="append", default=[],
                    help="explicit tracker base URL(s); skips the yaml")
    ap.add_argument("--write", action="store_true", help="command a trim and read it back")
    ap.add_argument("--prn", type=int, help="PRN to poke (default: the first one listed)")
    ap.add_argument("--trim", type=float, default=0.02, help="chips to command (default 0.02)")
    a = ap.parse_args()

    if a.tracker:
        targets = {"(explicit)": a.tracker}
    else:
        targets = trackers_from_chains(a.chains_yaml, set(a.chain))
    flat = [(c, t) for c, ts in sorted(targets.items()) for t in ts]
    if not flat:
        raise SystemExit("no tracker endpoints found")
    print("%d tracker instance(s) over %d chain(s)" % (len(flat), len(targets)))

    bad, seen_paths, live = [], {}, []
    for chain, base in flat:
        base = base.rstrip("/")
        try:
            st = get(base + "/get_trim")
        except Exception as e:
            bad.append("%s %s: /get_trim failed (%s)" % (chain, base, e))
            continue
        # (1) the endpoint's presence is inferred from the FIELDS get_trim now serves. A node
        # on the old build answers /get_trim happily and has no set_trim at all, so checking
        # only that get_trim responds would pass on exactly the build this exists to catch.
        if "posts" not in st or "ttl_s" not in st:
            bad.append("%s %s: OLD BUILD -- /get_trim has no `posts`/`ttl_s`, so /set_trim "
                       "does not exist here" % (chain, base))
            continue
        # (3) the in-tracker loop must be off
        if st.get("enabled"):
            bad.append("%s %s: code_trim is ENABLED -- the in-tracker loop and the fleet loop "
                       "would both write trim[]" % (chain, base))
        if not st.get("ttl_s"):
            bad.append("%s %s: trim_ttl_s is 0 -- a trim from a dead controller would stand "
                       "forever" % (chain, base))
        live.append((chain, base, st))

    # (2) distinct endpoints. Two instances answering the same URL is the override trap.
    for chain, base, _st in live:
        seen_paths.setdefault(base, []).append(chain)
    for base, chains in seen_paths.items():
        if len(chains) > 1:
            bad.append("%s is listed for %d chains (%s) -- endpoints must be per instance"
                       % (base, len(chains), ", ".join(chains)))

    for chain, base, st in live:
        prns = [t["prn"] for t in st.get("trims", [])]
        print("  %-9s %-46s ttl %.1fs  posts %d  expired %d  %d PRN slots"
              % (chain, base, st.get("ttl_s", 0), st.get("posts", 0), st.get("expired", 0),
                 len(prns)))

    if a.write and live:
        print("\n--- WRITE TEST (trim %+.3f chips; trim_ttl_s zeroes it if this dies) ---"
              % a.trim)
        for chain, base, st in live:
            prn = a.prn or (st["trims"][0]["prn"] if st.get("trims") else None)
            if prn is None:
                bad.append("%s %s: no PRN slots to poke" % (chain, base))
                continue
            try:
                post(base + "/set_trim", [{"prn": prn, "trim_chips": a.trim, "win": 0}])
                back = get(base + "/get_trim")
                got = next((t["trim_chips"] for t in back["trims"] if t["prn"] == prn), None)
                ok = got is not None and abs(got - a.trim) < 1e-9
                grew = back.get("posts", 0) > st.get("posts", 0)
                print("  %-9s PRN %-3d wrote %+.3f  read %s  posts %d->%d  %s"
                      % (chain, prn, a.trim, ("%+.6f" % got) if got is not None else "MISSING",
                         st.get("posts", 0), back.get("posts", 0),
                         "ok" if (ok and grew) else "MISMATCH"))
                if not ok:
                    bad.append("%s %s PRN %d: wrote %+.6f, read back %s"
                               % (chain, base, prn, a.trim, got))
                if not grew:
                    bad.append("%s %s: `posts` did not increment -- the POST was not applied "
                               "by set_trim_callback" % (chain, base))
            except Exception as e:
                bad.append("%s %s: /set_trim failed (%s)" % (chain, base, e))
            finally:
                # RESTORE, always. The TTL would do it in a few seconds anyway; doing it here
                # means the instrument is not carrying a deliberate offset for even that long.
                try:
                    post(base + "/set_trim", [{"prn": prn, "trim_chips": 0.0, "win": 0}])
                except Exception:
                    pass
        time.sleep(0.2)

    if bad:
        print("\nFAIL -- %d problem(s):" % len(bad))
        for b in bad:
            print("  " + b)
        return 1
    print("\nPASS -- %d instance(s): /set_trim present and distinct, code_trim off, TTL set%s."
          % (len(live), ", value read back" if a.write else " (read-only, add --write)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
