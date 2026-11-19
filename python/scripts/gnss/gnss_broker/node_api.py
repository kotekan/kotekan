"""A chain instance's node REST requests, at its row's own path or by chain.

The broker names a chain instance by its row's REST prefix, http://<node>:<port>/gnss<g>_<chain>_
<stage> for the stages inject, n2assemble and n2combine, as the fleet expansion lists it
(config/gnss_fleet.py). A node generated with --inject-host, --record-host or --combine-host
hosts that stage's rows in one stage per GPU half, gnss<g>_inject, gnss<g>_records or
gnss<g>_combine, which serves every row's verbs by chain: a GET at /gnss<g>_<host>/<verb>?chain=
<chain>, a POST there of {"chain": <chain>, "body": <the row's own request>}, each answered with
the row's own reply. The rows' own paths stay registered beside them until they are removed.

url() (a GET) and post_request() put a request in the form its node serves: by chain where the
half's host is listed (set_hosts(), the broker's --node-hosts, or a tool's own list from the
node's generated config), and at the row's own path otherwise. A prefix that is not a row of
those stages named by its broker chain (a search, a tap, a name generated before the tags) is
used as given.

canonical() names a request alike in either form, for the transcript gate (broker_equiv.py).
"""
import json
import os
import re
import sys
import urllib.parse

from . import signals

#: The per-chain stages a GPU half's host serves by chain, and that host: gnss<g>_<host>.
HOSTS = {"inject": "inject", "n2assemble": "records", "n2combine": "combine"}
_HOST_STAGE = {h: s for s, h in HOSTS.items()}
_ROW = re.compile(
    r"^(https?://[^/?#]+)/gnss(\d+)_([a-z0-9]+_[a-z0-9]+)_(inject|n2assemble|n2combine)$"
)
_HOST_PATH = re.compile(r"^/gnss(\d+)_(inject|records|combine)/([a-z0-9_]+)$")
_ROW_PATH = re.compile(
    r"^/(gnss(\d+)(?:_[a-z0-9_]+?)?_(inject|n2assemble|n2combine))/([a-z0-9_]+)$"
)


_HOST_BASE = re.compile(r"^https?://[^/?#]+/gnss\d+_(inject|records|combine)$")
_hosts = frozenset()


def set_hosts(bases):
    """Ask by chain on the GPU halves whose hosts are listed, as their bases
    http://<node>:<port>/gnss<g>_<host> (gnss_fleet.Fleet.host_bases), and at the rows' own
    paths everywhere else. Process-wide, as the hosts are the fleet's for every chain. A
    ValueError names any base that is not a half's host."""
    global _hosts
    want = frozenset(b.strip().rstrip("/") for b in bases if b.strip())
    bad = sorted(b for b in want if not _HOST_BASE.match(b))
    if bad:
        raise ValueError("not a GPU half's host: %s" % ", ".join(bad))
    _hosts = want


def hosts():
    return _hosts


def row(prefix):
    """(origin, half, chain, stage) of a row's REST prefix named by its broker chain, else None."""
    m = _ROW.match(prefix)
    if not m or m.group(3) not in signals._CHAINS:
        return None
    return m.group(1), int(m.group(2)), m.group(3), m.group(4)


def host_base(origin, half, stage):
    """The base of the host that serves `stage` by chain on a GPU half."""
    return "%s/gnss%d_%s" % (origin, half, HOSTS[stage])


def _hosted(prefix, hosts):
    hosts = _hosts if hosts is None else hosts
    r = row(prefix) if hosts else None
    return r if r and host_base(r[0], r[1], r[3]) in hosts else None


def url(prefix, verb, query="", hosts=None):
    """The URL of a GET of `verb` of the instance at `prefix`, `query` ("window=3") appended.
    `hosts`, the bases of the hosts that serve by chain, defaults to set_hosts()'."""
    r = _hosted(prefix, hosts)
    if r is None:
        return "%s/%s%s" % (prefix, verb, "?" + query if query else "")
    return "%s/%s?chain=%s%s" % (
        host_base(r[0], r[1], r[3]),
        verb,
        r[2],
        "&" + query if query else "",
    )


def post_request(prefix, verb, payload, hosts=None):
    """(url, body) of a POST of `payload` to `verb` of the instance at `prefix` (`hosts` as for
    url())."""
    r = _hosted(prefix, hosts)
    if r is None:
        return "%s/%s" % (prefix, verb), payload
    return (
        "%s/%s" % (host_base(r[0], r[1], r[3]), verb),
        {"chain": r[2], "body": payload},
    )


_legacy = []  # gnss_fleet.stage_chain, imported on first use


def _legacy_chain(name, stage):
    """A row name generated before the tags (gnss_fleet.stage_chain's allowlists), or None."""
    if not _legacy:
        conf = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 4, "config"
        )
        conf = os.path.normpath(conf)
        if conf not in sys.path:
            sys.path.append(conf)
        import gnss_fleet

        _legacy.append(gnss_fleet.stage_chain)
    return _legacy[0](name, stage)


def _key(origin, half, stage, chain, verb, query):
    q = urllib.parse.urlencode(sorted(query))
    return "node:%s/gnss%d/%s/%s/%s%s" % (
        origin,
        half,
        stage,
        chain,
        verb,
        "?" + q if q else "",
    )


def canonical(url, payload=None):
    """(key, payload) of a request, the same whether it went to a row's own path or to its host
    by chain: a row's request is node:<origin>/gnss<g>/<stage>/<chain>/<verb>[?<query>] with the
    row's own body, its query less the chain, sorted. Any other request is (url, payload) as
    given. A host's path without a chain is not a row's request."""
    u = urllib.parse.urlsplit(url)
    origin = "%s://%s" % (u.scheme, u.netloc)
    query = urllib.parse.parse_qsl(u.query, keep_blank_values=True)
    m = _HOST_PATH.match(u.path)
    if m:
        chains = [v for k, v in query if k == "chain"]
        rest = [(k, v) for k, v in query if k != "chain"]
        stage = _HOST_STAGE[m.group(2)]
        if payload is None and len(chains) == 1:
            return (
                _key(origin, int(m.group(1)), stage, chains[0], m.group(3), rest),
                None,
            )
        if (
            payload is not None
            and not chains
            and isinstance(payload, dict)
            and set(payload) == {"chain", "body"}
        ):
            return (
                _key(
                    origin, int(m.group(1)), stage, payload["chain"], m.group(3), rest
                ),
                payload["body"],
            )
    m = _ROW_PATH.match(u.path)
    if m:
        r = row(origin + "/" + m.group(1))
        chain = r[2] if r else _legacy_chain(m.group(1), m.group(3))
        if chain:
            return (
                _key(origin, int(m.group(2)), m.group(3), chain, m.group(4), query),
                payload,
            )
    return url, payload


def canonical_text(url, payload=None):
    """canonical() as the transcript gate hashes it: (key, the payload's sorted JSON)."""
    k, p = canonical(url, payload)
    return k, json.dumps(p, sort_keys=True)


def canonical_requests(url, payload=None):
    """[(key, payload)] of a request as canonical() names it, or, for an inject host's batched
    set_trim ({"chains": {<chain>: <request>, ...}}), one per chain in chain order, each as the
    chain's own set_trim, so a gather's batched POSTs compare with its per-chain ones."""
    u = urllib.parse.urlsplit(url)
    m = _HOST_PATH.match(u.path)
    if (
        m
        and m.group(2) == "inject"
        and m.group(3) == "set_trim"
        and isinstance(payload, dict)
        and list(payload) == ["chains"]
        and isinstance(payload["chains"], dict)
    ):
        origin = "%s://%s" % (u.scheme, u.netloc)
        return [
            (_key(origin, int(m.group(1)), "inject", c, "set_trim", []), b)
            for c, b in sorted(payload["chains"].items())
        ]
    return [canonical(url, payload)]
