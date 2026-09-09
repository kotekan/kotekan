#!/usr/bin/env python3
"""Fetch a running kotekan pipeline's graph and render it.

Wraps the ``/pipeline_dot`` endpoint (see lib/core/kotekanMode.cpp): fetches the
graph from a running kotekan, hands it to graphviz, and writes an image. The
graph is drawn from the live pipeline, so it carries what the config alone
cannot -- exact producer/consumer directions, real array layouts, measured
rates, and which buffers are backed up.

Examples
--------
    # render the pipeline on this machine
    tools/pipeline_graph.py -o pipeline.svg

    # structure only, no GPU internals, from another host
    tools/pipeline_graph.py -H gpu-node-3 --no-runtime --no-kernels -o structure.pdf

    # keep an eye on where frames are piling up
    tools/pipeline_graph.py --watch 5 -o pipeline.svg

    # the raw DOT, to edit or to diff between runs
    tools/pipeline_graph.py --no-runtime -o pipeline.dot

Only the standard library is needed, plus the `dot` executable for any output
format other than .dot (and .json, which is fetched from /pipeline_json).
"""

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def fetch(host: str, port: int, params: dict, as_json: bool, timeout: float) -> bytes:
    """GET the graph from a running kotekan."""
    endpoint = "pipeline_json" if as_json else "pipeline_dot"
    query = urllib.parse.urlencode(params)
    url = f"http://{host}:{port}/{endpoint}" + (f"?{query}" if query else "")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"{url}: HTTP {e.code} {e.reason}")
    except urllib.error.URLError as e:
        sys.exit(f"{url}: {e.reason} (is kotekan running with a pipeline started?)")


def render(dot_text: bytes, out_path: Path) -> None:
    """Run the graph through graphviz into out_path, whose suffix picks the format."""
    fmt = out_path.suffix.lstrip(".") or "svg"
    try:
        subprocess.run(
            ["dot", f"-T{fmt}", "-o", str(out_path)], input=dot_text, check=True
        )
    except FileNotFoundError:
        sys.exit(
            "graphviz 'dot' not found; install graphviz, or write a .dot file instead"
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"dot failed: {e}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and render the graph of a running kotekan pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-H", "--host", default="localhost", help="kotekan host (default localhost)"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=12048, help="REST port (default 12048)"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output file; the extension picks the format (.svg/.png/.pdf/.dot/.json). "
        "Default: DOT to stdout",
    )
    parser.add_argument(
        "--rankdir",
        default=None,
        choices=["LR", "TB", "RL", "BT"],
        help="layout direction (kotekan defaults to LR)",
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="do not group stages by config section",
    )
    parser.add_argument("--no-legend", action="store_true", help="omit the colour key")
    parser.add_argument(
        "--no-pools", action="store_true", help="omit the metadata pool list"
    )
    parser.add_argument(
        "--no-kernels",
        action="store_true",
        help="hide GPU commands and device memory (much smaller graph on a GPU pipeline)",
    )
    parser.add_argument(
        "--no-runtime",
        action="store_true",
        help="structure only: no measured rates, fill levels or state colouring. Use this "
        "to compare two runs, since the live figures differ on every fetch",
    )
    parser.add_argument(
        "--no-urls", action="store_true", help="no links on buffer nodes"
    )
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="re-fetch and re-render on this interval until interrupted",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default 10)",
    )
    args = parser.parse_args(argv)

    params = {}
    for flag, name in (
        (args.no_cluster, "cluster"),
        (args.no_legend, "legend"),
        (args.no_pools, "pools"),
        (args.no_kernels, "kernels"),
        (args.no_runtime, "runtime"),
        (args.no_urls, "urls"),
    ):
        if flag:
            params[name] = 0
    if args.rankdir:
        params["rankdir"] = args.rankdir

    out_path = Path(args.output) if args.output else None
    as_json = out_path is not None and out_path.suffix == ".json"

    if args.watch and out_path is None:
        sys.exit("--watch needs an output file (-o)")

    while True:
        graph = fetch(args.host, args.port, params, as_json, args.timeout)
        if out_path is None:
            sys.stdout.write(graph.decode())
        elif out_path.suffix in (".dot", ".json"):
            out_path.write_bytes(graph)
            print(f"wrote {out_path}", file=sys.stderr)
        else:
            render(graph, out_path)
            print(f"wrote {out_path}", file=sys.stderr)

        if not args.watch:
            return 0
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
