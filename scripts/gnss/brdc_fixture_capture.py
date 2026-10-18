#!/usr/bin/env python3
"""Freeze a set of IGS station-hourly files so the merge can be tested byte for byte offline.

    scripts/gnss/brdc_fixture_capture.py [outdir]      # default $GNSS_FIXTURES/brdc_merge_golden

The broker_equiv digest gate cannot see the ephemeris supply: replays pin the sky through
GNSS_BRDC_DIR, so the fetch and merge path never runs. The supply has its own policy surface
(mirrors, station list, per-constellation coverage targets, two cache TTLs, a negative cache,
a freshness window and a rolling store), and its output needs its own gate.

This captures the inputs (real station files, real RINEX quirks) and pins the clock, so
gnss_broker/test_brdc_golden.py can run the whole merge hermetically and hash the result.

The clock is part of the fixture: the merge prunes and counts against `when`, so a capture
without a pinned timestamp is not reproducible. It goes in the manifest.
"""
import gzip
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone, timedelta

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "python",
        "scripts",
        "gnss",
    ),
)
import gnss_brdc_supply as sup  # noqa: E402

# Enough stations to exercise the coverage exit test (some carry no BeiDou at all) without
# making the fixture heavy. Two hours: the walk always attempts the two newest.
N_STATIONS = 6
N_HOURS = 2


def main():
    out = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.environ.get("GNSS_FIXTURES", "/home/kvand/gnss/fixtures"),
            "brdc_merge_golden",
        )
    )
    os.makedirs(out, exist_ok=True)
    tok = sup._earthdata_token()
    # Capture the hours before the current one: the current hour is published nowhere yet,
    # and a fixture of 404s tests nothing.
    when = datetime.now(timezone.utc).replace(minute=30, second=0, microsecond=0)
    got, manifest = 0, {"when": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "files": []}
    for dh in range(1, N_HOURS + 1):
        h = when - timedelta(hours=dh)
        for st in sup._HOURLY_STATIONS[:N_STATIONS]:
            name = None
            for url, hdrs in sup._hourly_sources(st, h, tok):
                try:
                    with urllib.request.urlopen(
                        urllib.request.Request(url, headers=hdrs), timeout=25
                    ) as r:
                        raw = r.read()
                    if raw[:2] != b"\x1f\x8b":
                        continue
                    name = url.rsplit("/", 1)[-1]
                    sup._atomic_write_bytes(os.path.join(out, name), raw)
                    n = len(
                        gzip.decompress(raw).decode("ascii", "replace").splitlines()
                    )
                    manifest["files"].append(
                        {
                            "name": name,
                            "station": st,
                            "hour": h.hour,
                            "bytes": len(raw),
                            "lines": n,
                        }
                    )
                    got += 1
                    print("  %-42s %6d B  %5d lines" % (name, len(raw), n))
                    break
                except Exception as e:
                    print("  %-42s -- %s" % (st, str(e)[:50]))
    if got < 2:
        print(
            "REFUSING to write a manifest for %d file(s): a fixture that captured nothing "
            "would make the golden test pass vacuously." % got
        )
        return 1
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    total = sum(x["bytes"] for x in manifest["files"])
    print(
        "\n%d file(s), %.0f kB, when=%s -> %s"
        % (got, total / 1024.0, manifest["when"], out)
    )
    print(
        "NEXT: run gnss_broker/test_brdc_golden.py --print-digest and paste the value into "
        "the GOLDEN constant in that file (it lives in the test, under review, not in the "
        "manifest -- a golden regenerated alongside its fixture can never go red)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
