#!/usr/bin/env python3
"""bffs's current bad-input list, in the shape a kotekan node's /updatable_config/bad_inputs holds.

    python3 config/bffs_bad_inputs.py                 # print the update body (JSON)
    python3 config/bffs_bad_inputs.py --out FILE      # write it, for a POST

bffs (the observatory's bad-feed flagger, on choco) keeps its list as FEED LABELS in
/var/lib/choco/bffs/state.json and POSTs their POSITIONS in the label list it keeps beside them --
the flat [P][D] input index -- with start_time = its update time plus choco's sync_delay
(/etc/choco/bffs.yaml). choco relays that to every node in its cx group, except nodes in
maintenance mode, which ours are: choco would also push configs we must not have overwritten. So
we rebuild the identical body here, for two consumers that must never disagree:

  * config/gen_chord_gnss_config.py injects it into every node config it writes, so a node starts
    on the current list (a node down when bffs changed it would otherwise start on a stale one);
  * scripts/gnss/bad_inputs_cron.sh POSTs it to the running nodes whenever bffs changes it.

Identical means byte-identical to what a stock node holds: scripts/gnss/stock_parity.py --live
compares it.
"""
import argparse
import json
import subprocess
import sys

import yaml

STATE = "choco:/var/lib/choco/bffs/state.json"
CONF = "choco:/etc/choco/bffs.yaml"


def read(path, timeout=20.0):
    """`host:/path` over ssh (BatchMode), or a local path."""
    host, _, remote = path.partition(":")
    if remote:
        return subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                host,
                "cat " + remote,
            ],
            capture_output=True,
            timeout=timeout,
            check=True,
        ).stdout.decode()
    with open(path) as fh:
        return fh.read()


def update_body(state, conf):
    """The POST body: {"bad_inputs": [index...], "start_time": float, "update_id": str}.

    Exactly the updatable block's keys bar kotekan_update_endpoint, because kotekan's
    configUpdater rejects an update with a key missing, a key extra, or a type changed
    (start_time must stay a float).
    """
    labels, bad = list(state["labels"]), list(state["bad_inputs"])
    missing = [lb for lb in bad if lb not in labels]
    if missing:
        raise ValueError(
            "bffs flags labels that are not in its own label list: %s" % missing[:5]
        )
    delay = float(((conf or {}).get("choco") or {}).get("sync_delay", 0.0))
    return {
        "bad_inputs": sorted(labels.index(lb) for lb in bad),
        "start_time": float(state["updated"]) + delay,
        "update_id": str(state["update_id"]),
    }


def fetch(state_path=STATE, conf_path=CONF, timeout=20.0):
    """(body, state_path) from bffs's own files. Raises on any failure: callers decide."""
    state = json.loads(read(state_path, timeout))
    conf = yaml.safe_load(read(conf_path, timeout))
    return update_body(state, conf), state_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--state", default=STATE, help="bffs state.json (host:/path or local)"
    )
    ap.add_argument("--conf", default=CONF, help="bffs.yaml, for sync_delay")
    ap.add_argument("--out", default=None, help="write the body here instead of stdout")
    a = ap.parse_args()
    body, _ = fetch(a.state, a.conf)
    text = json.dumps(body)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    sys.stderr.write(
        "bffs: %d bad inputs, %s\n" % (len(body["bad_inputs"]), body["update_id"])
    )


if __name__ == "__main__":
    main()
