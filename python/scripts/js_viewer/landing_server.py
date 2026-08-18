#!/usr/bin/env python3
"""landing_server.py -- tiny static server for the hub landing page.

Serves ``chooser.html`` as ``/`` (plus ``sources.json`` and any sibling asset
the chooser references) so people can hit ``http://<host>/`` and land straight
on the source picker instead of ``/chooser.html`` on a per-source port.

Unlike the per-source ``livebeam_server`` instances, this holds NO kotekan
connection -- it's a pure static file server, so the landing page is up
regardless of which sources are currently streaming. It runs unprivileged on a
high port (default 8090); port 80 reaches it via a firewalld forward-port
(REDIRECT) rule -- see ARO_OPERATIONS.md. (It can also bind 80 directly if run
as root with ``--port 80``, but the redirect keeps this process unprivileged.)

The per-source viewer/`/status` links in chooser.html point at each source's own
http_port (8080/8081/...), not at this server, so this only ever needs to serve
the chooser and sources.json.
"""
import argparse
import functools
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler


class LandingHandler(SimpleHTTPRequestHandler):
    # Bare "/" serves the chooser (the whole point of this server), rather than
    # the directory index (index.html, which is the viewer). Everything else is
    # served literally from the static root.
    def _reroot(self):
        if self.path.split("?", 1)[0] in ("/", ""):
            self.path = "/chooser.html"

    def do_GET(self):
        self._reroot()
        return super().do_GET()

    def do_HEAD(self):
        self._reroot()
        return super().do_HEAD()

    def log_message(self, fmt, *args):  # quiet; the respawn wrapper logs lifecycle
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8090,
                    help="TCP port to serve on (default 8090; 80 needs root)")
    ap.add_argument("--bind", default="0.0.0.0", help="address to bind (default all)")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="static root (default: this script's dir)")
    args = ap.parse_args()

    handler = functools.partial(LandingHandler, directory=args.root)
    httpd = HTTPServer((args.bind, args.port), handler)
    print("landing server on {}:{}  root={}  (/ -> chooser.html)".format(
        args.bind, args.port, args.root), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
