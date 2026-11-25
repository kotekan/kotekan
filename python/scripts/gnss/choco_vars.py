"""Production's template variables for the stock render.

choco renders config/chord/pathfinder.j2 for every node with /etc/choco/configs/vars.yaml, so a
render of the stock template that is to match production needs the same variables. Without them
dtv.j2 falls back to its detect-only DTV bundle, and choco's /dtv_mask_enabled push then stops
the node (#244). Read over ssh like the EOP table (gen_chord_gnss_config.choco_eop_table).

env:  GNSS_STOCK_VARS  host:path or a local file to read instead, or "none" for the template's
                       defaults (offline renders and tests only)
"""
import os
import subprocess

import yaml

SOURCE = "choco:/etc/choco/configs/vars.yaml"
_cache = {}


def load(source=None, timeout=10.0):
    """The variables as a dict. Raises when they cannot be read: no render is better than one
    that production's pushes can stop."""
    src = source or os.environ.get("GNSS_STOCK_VARS") or SOURCE
    if src == "none":
        return {}
    if src not in _cache:
        if os.path.exists(src) or ":" not in src:
            with open(src) as fh:
                text = fh.read()
        else:
            host, _, path = src.partition(":")
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", host, "cat " + path],
                capture_output=True,
                timeout=timeout,
            )
            if r.returncode:
                raise RuntimeError(
                    "production's template variables unreadable from %s: %s"
                    % (src, r.stderr.decode().strip())
                )
            text = r.stdout.decode()
        v = yaml.safe_load(text) or {}
        if not isinstance(v, dict):
            raise RuntimeError("%s is not a mapping of template variables" % src)
        _cache[src] = v
    return dict(_cache[src])
