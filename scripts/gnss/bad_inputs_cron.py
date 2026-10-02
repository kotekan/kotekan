#!/usr/bin/env python3
"""Keep the running GNSS nodes' bad-input list on bffs's. Cron, on the gnss VM.

    */5 * * * *  /home/kvand/gnss/kotekan/scripts/gnss/bad_inputs_cron.py

WHY. choco relays bffs's list to every node in its cx group EXCEPT nodes in maintenance mode, and
ours are in maintenance mode: choco would also push configs we must not have overwritten. So a bffs
change reaches the stock nodes and not ours, and our bad-feed mask -- folded into N2Accumulate and
sent to recv1 beside the N^2 -- stays stale until a node restarts on a regenerated config. The
same gap eop_cron.sh closes for the EOP table, here at bffs's cadence rather than hourly.

EACH RUN (cheap when nothing changed):
  1. stage bffs's state.json + bffs.yaml from choco only when state.json changed (mtime and size,
     by ssh stat: the file carries ~2.5 MB of history), and build the update body with
     config/bffs_bad_inputs.py -- the translation the generator uses, so a restarted node and a
     running one cannot disagree;
  2. read each node's live config: only a node that answers and runs a GNSS config (gnssN_*
     stages) qualifies. A stock node or a down one is skipped, never pushed to;
  3. POST the body to nodes whose live block differs, read the block back, and check that
     bufferBadInputs did not drop it as out of order. That rejection is silent at the REST layer:
     configUpdater stores the values either way, so only the stage's late-update counter shows it.

env: GNSS_NODES (default the six), BADIN_LOG (default /var/tmp/gnss-logs/bad_inputs_cron.log),
     DRY_RUN=1 (report, push nothing), FORCE=1 (push even when current: a format test).
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(K, "config"))
import bffs_bad_inputs  # noqa: E402

NODES = os.environ.get("GNSS_NODES", "cx19 cx27 cx42 cx43 cx44 cx51").split()
LOG = os.environ.get("BADIN_LOG", "/var/tmp/gnss-logs/bad_inputs_cron.log")
STAGE = "/var/tmp/gnss-logs/bffs-state.json"
STAGE_CONF = "/var/tmp/gnss-logs/bffs.yaml"
STAGE_META = STAGE + ".meta"
DRY = os.environ.get("DRY_RUN") == "1"
FORCE = os.environ.get("FORCE") == "1"
LATE = "kotekan_bufferbadinputs_late_update_count"


def log(msg):
    with open(LOG, "a") as fh:
        fh.write(msg + "\n")


def http(url, body=None, timeout=10):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()


def stage():
    """Refresh the staged bffs files if choco's state.json changed. Returns a note for the log."""
    host, _, path = bffs_bad_inputs.STATE.partition(":")
    try:
        meta = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                               "stat -c '%Y %s' " + path], capture_output=True, timeout=30,
                              check=True).stdout.decode().strip()
    except Exception as e:
        return "choco unreachable (%s); using the staged list" % type(e).__name__
    have = open(STAGE_META).read().strip() if os.path.exists(STAGE_META) else ""
    if meta == have and os.path.exists(STAGE) and os.path.exists(STAGE_CONF):
        return None
    for src, dst in ((bffs_bad_inputs.STATE, STAGE), (bffs_bad_inputs.CONF, STAGE_CONF)):
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", src,
                        dst + ".new"], timeout=60, check=True)
        os.replace(dst + ".new", dst)
    with open(STAGE_META, "w") as fh:
        fh.write(meta + "\n")
    return "staged bffs state (%s)" % meta


def late_count(n):
    _, text = http("http://%s:12048/metrics" % n, timeout=15)
    return sum(float(ln.split()[1]) for ln in text.splitlines() if ln.startswith(LATE + "{"))


def same(live, body):
    return (isinstance(live, dict) and live.get("update_id") == body["update_id"]
            and sorted(live.get("bad_inputs") or []) == body["bad_inputs"]
            and abs(float(live.get("start_time") or 0.0) - body["start_time"]) < 1e-3)


def main():
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        note = stage()
        state = json.load(open(STAGE))
        conf = bffs_bad_inputs.yaml.safe_load(open(STAGE_CONF))
        body = bffs_bad_inputs.update_body(state, conf)
    except Exception as e:
        log("== %s bad_inputs_cron: NO LIST (%s: %s); nothing pushed" % (now, type(e).__name__, e))
        return 1
    lines, fail, current = [], 0, 0
    for n in NODES:
        try:
            _, text = http("http://%s:12048/config" % n)
            cfg = json.loads(text)
        except Exception:
            lines.append("  %s skip (DOWN)" % n)
            continue
        if not isinstance(cfg, dict) or ("code" in cfg and len(cfg) <= 2):
            lines.append("  %s skip (DOWN)" % n)
            continue
        if not any(re.match(r"gnss\d_", k) for k in cfg):
            lines.append("  %s skip (NOT-GNSS)" % n)
            continue
        live = (cfg.get("updatable_config") or {}).get("bad_inputs")
        if same(live, body) and not FORCE:
            current += 1
            continue
        was = live.get("update_id") if isinstance(live, dict) else None
        if DRY:
            lines.append("  %s WOULD PUSH %s -> %s" % (n, was, body["update_id"]))
            continue
        try:
            late0 = late_count(n)
            status, _ = http("http://%s:12048/updatable_config/bad_inputs" % n, body)
            _, text = http("http://%s:12048/config" % n)
            back = (json.loads(text).get("updatable_config") or {}).get("bad_inputs")
            late1 = late_count(n)
        except Exception as e:
            lines.append("  %s FAIL (%s: %s)" % (n, type(e).__name__, e))
            fail += 1
            continue
        if status != 200 or not same(back, body):
            lines.append("  %s FAIL (POST %s, readback %s)" % (n, status, back and back.get("update_id")))
            fail += 1
        elif late1 > late0:
            lines.append("  %s DROPPED by bufferBadInputs as out of order (late %d -> %d): a newer"
                         " update is queued there" % (n, late0, late1))
            fail += 1
        else:
            lines.append("  %s PUSHED %s -> %s (%d bad)" % (n, was, body["update_id"],
                                                             len(body["bad_inputs"])))
    head = "== %s bad_inputs_cron %s (%d bad)%s: %d current" % (
        now, body["update_id"], len(body["bad_inputs"]), " DRY" if DRY else "", current)
    log(head + ("; " + note if note else ""))
    for ln in lines:
        log(ln)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
