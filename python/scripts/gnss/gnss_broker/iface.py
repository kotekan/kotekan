"""The kotekan GNSS interface: what a kotekan binary and this broker must agree on, and the copy
of it the broker speaks.

Every kotekan binary describes, from its own compiled constants, what its GNSS stages exchange
with programs outside kotekan: the signal descriptors, the telemetry header, the record slots and
what each holds, the cube, GPU-chain and metadata layouts, and a version per REST family
(lib/stages/gnss/gnssInterface.hpp). It serves the document at GET /interfaces/gnss, its digests
in GET /version under "interfaces", and both from `kotekan --interfaces-json`, which needs no
config (lib/core/interfaceRegistry.hpp).

The broker carries one such document, the golden (iface_golden.json beside this file), pinned to
the kotekan commit it was taken from (iface_pin.json). It is the protocol this broker speaks:
signals.py, telem.py, config/gnss_record_layout.py and config/chord_band_plan.py take their
constants from it, so the broker needs no kotekan tree to run. Moving the pin is a broker commit:

    python3 -m gnss_broker.iface sync --from <kotekan tree | http://host:port | dump file>

Canonical form, schema 1 (the C++ registry's): objects with printable-ASCII keys, arrays, int64
integers, booleans and printable-ASCII strings. There are no floats and no null: every double is
the string "%.17g" (1176.45e6 is "1176450000"). The canonical bytes are
json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True). The document's digest is
"fnv1a64:" and 16 lowercase hex digits of FNV-1a 64 over the canonical bytes of
{"interface": "gnss", "schema": 1, "sections": {...}}; a section's digest is the same over the
section's own canonical bytes. Changing the form means a new schema number.

compare() checks a binary's document against the golden, leaf by leaf:

    MATCH         every golden leaf in scope is there and equal, and nothing else is
    COMPATIBLE    every golden leaf in scope is there and equal, and the binary has more
                  (an added descriptor or constant)
    MISMATCH      a golden leaf is missing or different; the paths, with both values
    UNCOMPARABLE  not a gnss document of this schema: an error block, another schema, or a
                  body that does not hash to the digest the binary advertised
    LEGACY        a binary that serves no document (every one built before the registry)

A role scopes the comparison to what that program exchanges with the broker (ROLES), and the
signals scope narrows to the descriptors an origin's chains use.

Stdlib only: this runs in the broker (CPython 3.14t), the tools' venv, and /usr/bin/python3 for
the deploy preflights.

    python3 -m gnss_broker.iface show [SOURCE]
    python3 -m gnss_broker.iface diff A B [--role ROLE]
    python3 -m gnss_broker.iface sync --from SOURCE [--commit SHA]
    python3 -m gnss_broker.iface preflight --role ROLE [--config CFG] [--from SOURCE]
    python3 -m gnss_broker.iface check-fleet [CHAINS_YAML]

A SOURCE is `golden` (the vendored copy), a kotekan tree, http://host:port (a running kotekan),
a file holding a document or `kotekan --interfaces-json` output, or `-` for stdin.
"""
import argparse
import http.client
import json
import os
import re
import struct
import subprocess
import sys
import threading
import traceback
import urllib.error
import urllib.request

NAME = "gnss"
SCHEMA = 1
#: Every section of a schema 1 document. The golden must carry all of them.
SECTIONS = (
    "cube",
    "enums",
    "gpu_chain",
    "metadata",
    "record.chan",
    "record.cmb",
    "record.elem",
    "record.row",
    "rest",
    "signals",
    "telem",
)
#: What each program exchanges with the broker, as paths into the sections. "signals" can be
#: narrowed to the descriptors of the chains an origin serves.
ROLES = {
    "node": (
        "signals",
        "telem",
        "record.row",
        "record.chan",
        "rest/tracker",
        "rest/combiner",
        "rest/assembler",
        "rest/tap",
    ),
    "gather": ("telem", "record.row", "record.chan", "metadata", "rest/fleet_trim"),
    "aggregator": ("signals", "rest/search"),
    # The beam-cube archiver: the cube frames and their metadata, which the archive's readers
    # (gnss_cube_read.py) take apart.
    "cubearch": ("cube", "metadata"),
}
#: The sections that size frames: the generator stamps their digests into every config it
#: writes, and a binary must agree with the stamp before it starts on that config.
SIZING = ("telem", "record.row", "gpu_chain", "cube", "metadata")

MATCH = "MATCH"
COMPATIBLE = "COMPATIBLE"
MISMATCH = "MISMATCH"
UNCOMPARABLE = "UNCOMPARABLE"
LEGACY = "LEGACY"

#: `preflight`'s exit status for a refusal. Not 1 or 2: the interpreter exits 1 on an uncaught
#: exception or a module that does not import and 2 on a usage error, and neither may refuse a
#: start.
PREFLIGHT_REFUSED = 3

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(_HERE, "iface_golden.json")
PIN_PATH = os.path.join(_HERE, "iface_pin.json")
#: Where a kotekan tree keeps the document its build serves.
KOTEKAN_GOLDEN = "lib/stages/gnss/gnss_interface.json"
#: The chains yaml check-fleet reads by default: this checkout's.
CHAINS_YAML = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "config", "gnss_chains_chord.yaml")
)

_INT64_MIN, _INT64_MAX = -(1 << 63), (1 << 63) - 1


class GoldenError(RuntimeError):
    """The vendored golden is missing, unreadable, malformed, or not what its pin says."""


class SourceError(ValueError):
    """A document could not be read from where it was asked for."""


class Legacy(SourceError):
    """The binary serves no gnss interface document: it predates the registry."""


class Uncomparable(SourceError):
    """A gnss document that cannot be compared: an error block, another schema, or a body that
    does not hash to the digest served with it."""


# -- canonical form and digests --------------------------------------------------------------


def _check(value, path):
    """Raise ValueError naming the JSON pointer of the first value the form does not allow."""
    t = type(value)
    if t is dict:
        for k, v in value.items():
            if type(k) is not str or not (k.isascii() and k.isprintable()):
                raise ValueError("key in '%s' is not printable ASCII" % path)
            _check(v, path + "/" + k)
    elif t is list:
        for i, v in enumerate(value):
            _check(v, "%s/%d" % (path, i))
    elif t is str:
        if not (value.isascii() and value.isprintable()):
            raise ValueError("string at '%s' is not printable ASCII" % path)
    elif t is int:
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError("integer at '%s' does not fit in an int64" % path)
    elif t is float:
        raise ValueError(
            "float at '%s'; a double is written as a '%%.17g' string" % path
        )
    elif t is not bool:
        raise ValueError("%s at '%s' is not allowed" % (t.__name__, path))


def canonical(obj):
    """The canonical bytes of obj: the C++ registry's canonical_json, byte for byte."""
    _check(obj, "")
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def fnv1a64(data):
    """FNV-1a 64 of a bytes object, as an int."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def digest(obj):
    """'fnv1a64:<16 hex>' of bytes as they are, or of anything else's canonical bytes."""
    data = obj if isinstance(obj, (bytes, bytearray)) else canonical(obj)
    return "fnv1a64:%016x" % fnv1a64(data)


def doc_digest(doc):
    """The document digest kotekan serves in /version: over {interface, schema, sections}."""
    return digest(
        {
            "interface": doc["interface"],
            "schema": doc["schema"],
            "sections": doc["sections"],
        }
    )


def section_digests(doc):
    """{section: digest}, as /version serves them."""
    return {k: digest(v) for k, v in doc["sections"].items()}


def canonical_double(x):
    """How the document writes a double: '%.17g', which reads back to the same double."""
    return "%.17g" % x


def dbl(s):
    """A document double (a '%.17g' string) as a float. Anything else raises ValueError: a
    double written any other way is not one the C++ wrote."""
    if type(s) is not str:
        raise ValueError("%r is not a document double (a '%%.17g' string)" % (s,))
    v = float(s)
    if canonical_double(v) != s:
        raise ValueError(
            "%r is not written as '%%.17g' (%s)" % (s, canonical_double(v))
        )
    return v


def validate_doc(doc, require_sections=False):
    """Raise ValueError unless doc is a canonical gnss document of this schema."""
    if type(doc) is not dict:
        raise ValueError("not a JSON object")
    keys = sorted(doc)
    if keys != ["interface", "schema", "sections"]:
        raise ValueError(
            "top-level keys %s, want interface, schema and sections" % ", ".join(keys)
        )
    if doc["interface"] != NAME:
        raise ValueError("interface %r, not %r" % (doc["interface"], NAME))
    if doc["schema"] != SCHEMA or type(doc["schema"]) is not int:
        raise ValueError(
            "schema %r; this code reads schema %d only" % (doc["schema"], SCHEMA)
        )
    if type(doc["sections"]) is not dict:
        raise ValueError("sections is not an object")
    canonical(doc)
    if require_sections:
        missing = [s for s in SECTIONS if s not in doc["sections"]]
        if missing:
            raise ValueError("no section %s" % ", ".join(missing))


# -- wire structs ----------------------------------------------------------------------------

_STRUCT_CODES = {
    "u8": "B",
    "i8": "b",
    "u16": "H",
    "i16": "h",
    "u32": "I",
    "i32": "i",
    "u64": "Q",
    "i64": "q",
    "f32": "f",
    "f64": "d",
}
_FIELD_TYPE = re.compile(r"^(char|[uif]\d+)(?:\[(\d+)\])?$")


def field_code(f_type):
    """One field's struct code: 'u16' -> 'H', 'u16[8]' -> '8H', 'char[16]' -> '16s'."""
    m = _FIELD_TYPE.match(f_type)
    if not m or (m.group(1) != "char" and m.group(1) not in _STRUCT_CODES):
        raise ValueError("unknown field type %r" % f_type)
    n = m.group(2)
    if m.group(1) == "char":
        return "%ss" % (n or "1")
    return (n or "") + _STRUCT_CODES[m.group(1)]


def struct_format(table):
    """The little-endian struct format of a document wire struct ({"size", "fields": [[name,
    offset, size, type], ...]}). The fields must tile the struct in order, as the C++
    static_assert requires, and each field's size must be its type's; ValueError otherwise."""
    fmt, off = "<", 0
    for name, f_off, f_size, f_type in table["fields"]:
        if f_off != off:
            raise ValueError(
                "field %s at offset %d, the fields before it end at %d"
                % (name, f_off, off)
            )
        code = field_code(f_type)
        if struct.calcsize("<" + code) != f_size:
            raise ValueError(
                "field %s: %d bytes, but %s is %d"
                % (name, f_size, f_type, struct.calcsize("<" + code))
            )
        fmt += code
        off += f_size
    if off != table["size"]:
        raise ValueError(
            "the fields end at %d, the struct is %d" % (off, table["size"])
        )
    return fmt


# -- the golden ------------------------------------------------------------------------------


class Golden(object):
    """The document this broker speaks, checked against its pin. Treat it as read-only: one
    instance is shared by every caller of golden()."""

    __slots__ = ("doc", "sections", "digest", "section_digests", "pin", "path")

    def __init__(self, doc, pin, path):
        self.doc = doc
        self.sections = doc["sections"]
        self.digest = doc_digest(doc)
        self.section_digests = section_digests(doc)
        self.pin = pin
        self.path = path

    @property
    def commit(self):
        """The kotekan commit the golden was taken from."""
        return self.pin["kotekan_commit"]

    def const(self, section, name):
        """An integer constant, such as const("telem", "TELEM_VERSION")."""
        v = self.sections[section][name]
        if type(v) is not int:
            raise GoldenError("%s/%s is %r, not an integer" % (section, name, v))
        return v

    def slot(self, section, name):
        """A slot index, such as slot("record.row", "REC_P_RE")."""
        v = self.sections[section][name]["slot"]
        if type(v) is not int:
            raise GoldenError("%s/%s/slot is %r, not an integer" % (section, name, v))
        return v

    def struct_format(self, section, name):
        """The struct format of a wire struct, such as ("telem", "TelemHeader")."""
        return struct_format(self.sections[section][name])

    def __repr__(self):
        return "Golden(%s, kotekan %s)" % (self.digest, self.commit)


def load_golden(path=GOLDEN_PATH, pin_path=PIN_PATH):
    """Read and check a golden: a canonical gnss document of this schema with every section,
    whose digest is the one its pin names. GoldenError otherwise, naming the file."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        raise GoldenError("cannot read the interface golden: %s" % e)
    try:
        with open(pin_path) as fh:
            pin = json.load(fh)
        commit, want = pin["kotekan_commit"], pin["digest"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise GoldenError("cannot read the interface pin %s: %r" % (pin_path, e))
    try:
        doc = json.loads(raw.decode("ascii"))
        validate_doc(doc, require_sections=True)
    except ValueError as e:
        raise GoldenError("%s is not a gnss interface document: %s" % (path, e))
    g = Golden(doc, pin, path)
    if g.digest != want:
        raise GoldenError(
            "%s digests to %s, but %s pins %s (kotekan %s). Resync it rather than edit it: "
            "python3 -m gnss_broker.iface sync --from <kotekan tree>"
            % (path, g.digest, pin_path, want, commit)
        )
    return g


_GOLDEN = None
_GOLDEN_LOCK = threading.Lock()


def golden():
    """The vendored golden, loaded and checked once per process."""
    global _GOLDEN
    if _GOLDEN is None:
        with _GOLDEN_LOCK:
            if _GOLDEN is None:
                _GOLDEN = load_golden()
    return _GOLDEN


# -- comparing -------------------------------------------------------------------------------

_ABSENT = object()


class Comparison(object):
    """A verdict: state, fails [(path, golden value, doc value)], extras [path], and the reason
    when UNCOMPARABLE. A path is a tuple; path_str() writes it."""

    __slots__ = ("state", "fails", "extras", "reason")

    def __init__(self, state, fails=(), extras=(), reason=""):
        self.state = state
        self.fails = list(fails)
        self.extras = list(extras)
        self.reason = reason

    def __repr__(self):
        return "Comparison(%s, %d failing, %d extra)" % (
            self.state,
            len(self.fails),
            len(self.extras),
        )


def path_str(path):
    """('record.row', 'REC_CP', 'slot') -> 'record.row/REC_CP/slot'."""
    return "/".join(path)


def show_value(v):
    """A leaf for a message: JSON, or a short stand-in for a missing value or a container."""
    if v is _ABSENT:
        return "missing"
    if type(v) is dict:
        return "{...}" if v else "{}"
    if type(v) is list:
        s = json.dumps(v)
        return s if len(s) <= 60 else "[...%d]" % len(v)
    return json.dumps(v)


def scope(role, signals=None):
    """The paths (tuples into the sections) a role, or several roles together, is compared on.
    `signals`, an iterable of descriptor names, narrows the signals section to those
    descriptors."""
    roles = (role,) if isinstance(role, str) else tuple(role)
    for r in roles:
        if r not in ROLES:
            raise ValueError(
                "unknown role %r (known: %s)" % (r, ", ".join(sorted(ROLES)))
            )
    out = []
    for p in [p for r in roles for p in ROLES[r]]:
        t = tuple(p.split("/"))
        if t == ("signals",) and signals is not None:
            ts = [("signals", n) for n in sorted(set(signals))]
        else:
            ts = [t]
        out += [t for t in ts if t not in out]
    return out


def _get(node, path):
    for k in path:
        if type(node) is dict:
            node = node.get(k, _ABSENT)
        elif type(node) is list and k.isdigit() and int(k) < len(node):
            node = node[int(k)]
        else:
            return _ABSENT
        if node is _ABSENT:
            break
    return node


def _leaves(v, path):
    """(path, value) of every leaf under v; an empty container is a leaf."""
    if type(v) is dict and v:
        for k in sorted(v):
            yield from _leaves(v[k], path + (k,))
    elif type(v) is list and v:
        for i, x in enumerate(v):
            yield from _leaves(x, path + (str(i),))
    else:
        yield path, v


def _walk(g, d, path, fails, extras):
    if d is _ABSENT:
        fails.extend((p, v, _ABSENT) for p, v in _leaves(g, path))
        return
    tg = type(g)
    if tg is dict or tg is list:
        if type(d) is not tg:
            fails.append((path, g, d))
            return
        if tg is dict:
            for k in sorted(g):
                _walk(g[k], d.get(k, _ABSENT), path + (k,), fails, extras)
            extras.extend(path + (k,) for k in sorted(d) if k not in g)
        else:
            for i, x in enumerate(g):
                _walk(
                    x, d[i] if i < len(d) else _ABSENT, path + (str(i),), fails, extras
                )
            extras.extend(path + (str(i),) for i in range(len(g), len(d)))
    elif type(d) is not tg or d != g:
        # Type-strict: true and 1 are different values to the C++ and to the document.
        fails.append((path, g, d))


def compare(ref, doc, role=None, signals=None):
    """Compare a document with a reference (a Golden or a document). With no role, every
    section of the reference; otherwise the scope of the role, or of a list of roles (see
    scope()). A reference path that the reference itself lacks is a caller error
    (ValueError)."""
    ref_doc = ref.doc if isinstance(ref, Golden) else ref
    try:
        validate_doc(doc)
    except ValueError as e:
        return Comparison(UNCOMPARABLE, reason=str(e))
    if role is None:
        paths = [(k,) for k in sorted(ref_doc["sections"])]
    else:
        paths = scope(role, signals)
    fails, extras = [], []
    for p in paths:
        g = _get(ref_doc["sections"], p)
        if g is _ABSENT:
            raise ValueError("%s is not in the reference document" % path_str(p))
        _walk(g, _get(doc["sections"], p), p, fails, extras)
    state = MISMATCH if fails else COMPATIBLE if extras else MATCH
    return Comparison(state, fails, extras)


def describe_fails(fails, a="golden", b="binary", limit=None):
    """One line per failing leaf: 'path: golden X, binary Y'."""
    lines = [
        "%s: %s %s, %s %s" % (path_str(p), a, show_value(g), b, show_value(d))
        for p, g, d in fails
    ]
    if limit is not None and len(lines) > limit:
        lines = lines[:limit] + ["... and %d more" % (len(lines) - limit)]
    return lines


def summary(c, limit=8):
    """'same', 'compatible (extra: ...)' or 'flag day: ...' -- what `iface diff` prints."""
    if c.state == UNCOMPARABLE:
        return "uncomparable: %s" % c.reason
    if c.state == MATCH:
        return "same"
    if c.state == COMPATIBLE:
        ex = [path_str(p) for p in c.extras]
        more = " ... and %d more" % (len(ex) - limit) if len(ex) > limit else ""
        return "compatible (extra: %s%s)" % (", ".join(ex[:limit]), more)
    return "flag day: %s" % "; ".join(describe_fails(c.fails, "A", "B", limit))


# -- the config stamp -------------------------------------------------------------------------

_STAMP = re.compile(
    r"^# gnss_iface: (fnv1a64:[0-9a-f]{16}) pin (\S+) sizing((?: \S+=fnv1a64:[0-9a-f]{16})+)\s*$"
)


def stamp_line(g):
    """The header line the generator writes into every config, naming the golden it sized the
    frames from and the digest of each sizing section."""
    return "# gnss_iface: %s pin %s sizing %s" % (
        g.digest,
        g.commit,
        " ".join("%s=%s" % (s, g.section_digests[s]) for s in SIZING),
    )


def parse_stamp(text, max_lines=80):
    """{"digest", "pin", "sizing": {section: digest}} from a config's header comment, or None
    when it carries no stamp (every config written before the stamp existed)."""
    for i, line in enumerate(text.splitlines()):
        if i >= max_lines or (line and not line.startswith("#")):
            break
        m = _STAMP.match(line)
        if m:
            sizing = dict(kv.split("=", 1) for kv in m.group(3).split())
            return {"digest": m.group(1), "pin": m.group(2), "sizing": sizing}
    return None


_CONFIG_SIGNAL = re.compile(r"^\s*signal:\s*([A-Z][A-Z0-9_]*)\s*$", re.M)


def config_signals(text):
    """The descriptor names a generated config's stages resolve (`signal: NAME`)."""
    return sorted(set(_CONFIG_SIGNAL.findall(text)))


# -- sources ---------------------------------------------------------------------------------


class Source(object):
    """A document and where it came from: commit is the kotekan commit when known, raw the
    bytes when they were a golden-form file."""

    __slots__ = ("doc", "commit", "what", "raw")

    def __init__(self, doc, commit, what, raw=None):
        self.doc, self.commit, self.what, self.raw = doc, commit, what, raw


def _check_served(doc, claimed, what):
    """Uncomparable unless doc is a canonical gnss document hashing to what was served with
    it (claimed: {"digest", "sections"} as --interfaces-json and /version give them)."""
    try:
        validate_doc(doc)
    except ValueError as e:
        raise Uncomparable("%s: %s" % (what, e))
    got = doc_digest(doc)
    if claimed.get("digest") is not None and claimed["digest"] != got:
        raise Uncomparable(
            "%s: the document digests to %s, but %s was served with it"
            % (what, got, claimed["digest"])
        )
    sd = section_digests(doc)
    sections = claimed.get("sections")
    for k, v in sorted(sections.items() if isinstance(sections, dict) else ()):
        if sd.get(k) != v:
            raise Uncomparable(
                "%s: section %s digests to %s, but %s was served with it"
                % (what, k, sd.get(k), v)
            )


def _from_entry(entry, what):
    """The document out of an --interfaces-json entry ({"gnss": {...}})."""
    if not isinstance(entry, dict):
        raise Uncomparable("%s: the gnss entry is not an object" % what)
    if "error" in entry:
        raise Uncomparable(
            "%s: the binary's gnss document is in error: %s" % (what, entry["error"])
        )
    if "doc" not in entry:
        raise Uncomparable("%s: no doc in the gnss entry" % what)
    _check_served(entry["doc"], entry, what)
    return entry["doc"]


def _git(tree, *args):
    try:
        p = subprocess.run(
            ["git", "-C", tree] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except OSError:
        return None
    return p.stdout.strip() if p.returncode == 0 else None


# What a failed HTTP read can raise: a refused or timed-out connection (OSError), an HTTP status
# (URLError), a broken reply (HTTPException) or a body that is not JSON (ValueError).
_READ_ERRORS = (OSError, ValueError, urllib.error.URLError, http.client.HTTPException)


def _http_get(url, timeout):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        e.close()
        raise


def read_version(base, timeout):
    """A running kotekan's GET /version, as a dict. SourceError when it cannot be read or is
    not a JSON object."""
    try:
        ver = json.loads(_http_get(base + "/version", timeout).decode("utf-8"))
    except _READ_ERRORS as e:
        raise SourceError("%s/version: %s" % (base, e))
    if not isinstance(ver, dict):
        raise SourceError("%s/version: not a JSON object" % base)
    return ver


def version_entry(ver, what):
    """The gnss entry ({schema, digest, sections}) of a /version reply. Legacy when there is
    none; Uncomparable when it is in error or of another schema."""
    if "interfaces" not in ver:
        raise Legacy(
            "%s serves no interfaces (kotekan %s)" % (what, ver.get("git_commit_hash"))
        )
    entry = ver["interfaces"].get(NAME) if isinstance(ver["interfaces"], dict) else None
    if entry is None:
        raise Legacy("%s serves no %s interface" % (what, NAME))
    if not isinstance(entry, dict):
        raise Uncomparable("%s: the %s entry is not an object" % (what, NAME))
    if "error" in entry:
        raise Uncomparable(
            "%s: the gnss document is in error: %s" % (what, entry["error"])
        )
    if entry.get("schema") != SCHEMA or type(entry.get("schema")) is not int:
        raise Uncomparable(
            "%s: schema %r; this code reads schema %d only"
            % (what, entry.get("schema"), SCHEMA)
        )
    return entry


def fetch_doc(base, entry, timeout):
    """GET /interfaces/gnss, checked against the digests /version served with it (entry).
    SourceError when it cannot be read; Uncomparable when it does not hash to them or is not
    canonical."""
    try:
        body = _http_get(base + "/interfaces/" + NAME, timeout)
        doc = json.loads(body.decode("ascii"))
    except _READ_ERRORS as e:
        raise SourceError("%s/interfaces/%s: %s" % (base, NAME, e))
    _check_served(doc, entry, base)
    if canonical(doc) != body:
        raise Uncomparable("%s/interfaces/%s: the body is not canonical" % (base, NAME))
    return doc


# A kotekan_version of a build from a modified tree: versioneer's "<v>+<n>.g<hash>.dirty".
_DIRTY = re.compile(r"[.+-]dirty$")


def _from_url(base, timeout):
    ver = read_version(base, timeout)
    doc = fetch_doc(base, version_entry(ver, base), timeout)
    commit = ver.get("git_commit_hash")
    if _DIRTY.search(str(ver.get("kotekan_version") or "")):
        # Built from a modified tree, so the document it serves need not be its commit's
        # golden: no commit describes it, as for a tree whose golden is edited (read_source).
        return Source(doc, None, "%s (a dirty build of %s)" % (base, commit))
    return Source(doc, commit, base)


def _from_bytes(raw, what, commit=None):
    if not raw.strip():
        raise Legacy(
            "%s: empty; a binary that predates --interfaces-json prints nothing" % what
        )
    if raw.startswith(b"Invalid option"):
        # What kotekan prints on stdout for an option it does not know.
        raise Legacy("%s: the binary does not know --interfaces-json" % what)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise SourceError("%s: not JSON (%s)" % (what, e))
    if isinstance(obj, dict) and "interface" in obj:
        try:
            validate_doc(obj)
        except ValueError as e:
            raise Uncomparable("%s: %s" % (what, e))
        return Source(obj, commit, what, raw)
    if isinstance(obj, dict) and "interfaces" in obj and "git_commit_hash" in obj:
        raise SourceError(
            "%s: a /version reply holds digests only, not the document" % what
        )
    if isinstance(obj, dict):
        if NAME not in obj:
            raise Legacy("%s: no %s interface in it" % (what, NAME))
        return Source(_from_entry(obj[NAME], what), commit, what)
    raise SourceError("%s: not a JSON object" % what)


def read_source(spec, timeout=3.0, stdin=None):
    """A Source from `golden`, a kotekan tree, http://host:port, a file, or `-` (stdin).
    Raises Legacy, Uncomparable or SourceError; GoldenError for a bad golden."""
    if spec == "golden":
        g = golden()
        return Source(g.doc, g.commit, "golden")
    if spec.startswith("http://") or spec.startswith("https://"):
        return _from_url(spec.rstrip("/"), timeout)
    if spec == "-":
        stream = stdin if stdin is not None else sys.stdin.buffer
        return _from_bytes(stream.read(), "stdin")
    if os.path.isdir(spec):
        path = os.path.join(spec, KOTEKAN_GOLDEN)
        if not os.path.exists(path):
            raise SourceError("%s holds no %s" % (spec, KOTEKAN_GOLDEN))
        commit = _git(spec, "log", "-1", "--format=%H", "--", KOTEKAN_GOLDEN) or None
        if commit and _git(spec, "status", "--porcelain", "--", KOTEKAN_GOLDEN):
            commit = None  # edited since that commit: no commit describes it
        with open(path, "rb") as fh:
            return _from_bytes(fh.read(), path, commit)
    try:
        with open(spec, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        raise SourceError("%s: %s" % (spec, e))
    return _from_bytes(raw, spec)


def format_golden(doc):
    """A golden's bytes: the form kotekan's gnssInterface.hpp writes its golden in."""
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def _write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


# -- the command line ------------------------------------------------------------------------


def _cmd_show(a):
    s = read_source(a.source)
    doc = s.doc
    print("source   %s" % s.what)
    print("digest   %s" % doc_digest(doc))
    if a.source == "golden":
        print("pin      kotekan %s" % golden().commit)
    elif s.commit:
        print("commit   %s" % s.commit)
    print("schema   %d" % doc["schema"])
    for k, v in sorted(section_digests(doc).items()):
        n = len(doc["sections"][k]) if isinstance(doc["sections"][k], dict) else 0
        print("  %-12s %s  (%d keys)" % (k, v, n))
    return 0


def _cmd_diff(a):
    sa, sb = read_source(a.a), read_source(a.b)
    sig = a.signals.split(",") if a.signals else None
    c = compare(sa.doc, sb.doc, role=a.role, signals=sig)
    print(summary(c, limit=a.limit))
    return 1 if c.state in (MISMATCH, UNCOMPARABLE) else 0


def _cmd_sync(a):
    if a.source == "golden":
        raise SourceError(
            "sync takes a kotekan tree, a URL or a dump, not the golden itself"
        )
    s = read_source(a.source)
    try:
        validate_doc(s.doc, require_sections=True)
    except ValueError as e:
        raise SourceError("%s: %s" % (s.what, e))
    commit = a.commit or s.commit
    if not commit:
        raise SourceError(
            "%s names no kotekan commit (a dump, a dirty build, or a golden edited since its "
            "last commit): give --commit" % s.what
        )
    text = format_golden(s.doc)
    if s.raw is not None and s.raw.decode("ascii") != text:
        # Not the form kotekan writes: vendor it as kotekan would have.
        print("note: %s is not in golden form; writing it as kotekan does" % s.what)
    try:
        old = load_golden(a.golden, a.pin)
        print("was      %s (kotekan %s)" % (old.digest, old.commit))
        print("change   %s" % summary(compare(old, s.doc), limit=a.limit))
    except GoldenError as e:
        print("was      unreadable (%s)" % e)
    pin = {
        "digest": doc_digest(s.doc),
        "kotekan_commit": commit,
        "source": KOTEKAN_GOLDEN,
    }
    _write(a.golden, text)
    _write(a.pin, json.dumps(pin, indent=1, sort_keys=True) + "\n")
    print("now      %s (kotekan %s) from %s" % (pin["digest"], commit, s.what))
    return 0


def _cmd_preflight(a):
    """Exit 0: start it (MATCH or COMPATIBLE, or nothing to compare, which is warned);
    PREFLIGHT_REFUSED (3): refuse, with an ERROR line per reason. Any other status, 2 for a
    usage error, a bad golden or an internal error, means no check was made, which the caller
    treats as a warning, never a refusal.

    Refused: a binary that disagrees with the broker's golden on the role's scope; a config
    descriptor the binary does not resolve (it would stop at stage construction); a sizing
    section whose digest is not the one the config was generated for. Warned and started: a
    LEGACY binary, an UNCOMPARABLE or unreadable document, an unstamped config."""
    tag = "iface preflight (%s)" % a.role
    g = golden()
    cfg_text, stamp = None, None
    if a.config:
        try:
            with open(a.config) as fh:
                cfg_text = fh.read()
        except OSError as e:
            print("%s: WARN cannot read the config %s: %s" % (tag, a.config, e))
        else:
            stamp = parse_stamp(cfg_text)
            if stamp is None:
                print(
                    "%s: WARN %s carries no gnss_iface stamp (written before the stamp); "
                    "its frame sizes are not checked" % (tag, a.config)
                )
    try:
        s = read_source(a.source, timeout=a.timeout)
    except Legacy as e:
        print("%s: WARN %s binary, not checked: %s" % (tag, LEGACY, e))
        return 0
    except Uncomparable as e:
        print("%s: WARN %s, not checked: %s" % (tag, UNCOMPARABLE, e))
        return 0
    except SourceError as e:
        print(
            "%s: WARN could not read the binary's interface, not checked: %s" % (tag, e)
        )
        return 0
    refuse = []
    if a.signals is not None:
        sig = sorted(set(x for x in a.signals.split(",") if x))
    elif cfg_text is not None:
        sig = config_signals(cfg_text)
    else:
        sig = None
    if sig is not None and "signals" in ROLES[a.role]:
        have = s.doc["sections"].get("signals")
        have = have if isinstance(have, dict) else {}
        for n in sig:
            if n not in have:
                refuse.append(
                    "the config names descriptor %s, which this binary does not resolve"
                    % n
                )
            elif n not in g.sections["signals"]:
                print(
                    "%s: WARN descriptor %s is not in the broker's golden (pin %s): the "
                    "broker cannot run its chain" % (tag, n, g.commit)
                )
        sig = [n for n in sig if n in g.sections["signals"]]
    c = compare(g, s.doc, role=a.role, signals=sig)
    if c.state == MISMATCH:
        refuse += [
            "%s against the broker's golden (pin %s)" % (ln, g.commit)
            for ln in describe_fails(c.fails, "golden", "binary", limit=a.limit)
        ]
    if stamp is not None:
        sd = section_digests(s.doc)
        for sec, want in sorted(stamp["sizing"].items()):
            if sd.get(sec) != want:
                refuse.append(
                    "section %s is %s in the binary, but the config was sized for %s "
                    "(pin %s)" % (sec, sd.get(sec), want, stamp["pin"])
                )
    for ln in refuse:
        print("%s: ERROR %s" % (tag, ln))
    if refuse:
        print(
            "%s: REFUSED: the binary (%s, document %s) disagrees with the broker's golden "
            "(pin %s) or the config's stamp on %d point(s), listed above"
            % (tag, s.what, doc_digest(s.doc), g.commit, len(refuse))
        )
        return PREFLIGHT_REFUSED
    extra = " (extra: %d leaves)" % len(c.extras) if c.extras else ""
    print(
        "%s: %s %s%s%s"
        % (
            tag,
            c.state,
            doc_digest(s.doc),
            extra,
            ", config stamp agrees" if stamp else "",
        )
    )
    return 0


def _cmd_check_fleet(a):
    """One read of every origin the chains yaml names, as the broker's watcher reads them
    (iface_watch.py). Exit 1 when one is in MISMATCH; a single read decides here."""
    try:
        import yaml  # every interpreter that runs this has it (the venvs; apt's for /usr/bin)
    except ImportError:
        raise SourceError("check-fleet reads the chains yaml, which needs PyYAML")
    from . import iface_watch

    try:
        with open(a.chains) as fh:
            cfg = yaml.safe_load(fh) or {}
        lines, bad = iface_watch.check_fleet(cfg, timeout=a.timeout)
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise SourceError("%s: %s" % (a.chains, e))
    print("\n".join(lines))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python3 -m gnss_broker.iface",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("show", help="digests and sections of a document")
    p.add_argument("source", nargs="?", default="golden")
    p = sub.add_parser("diff", help="same / compatible / flag day, B against A")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument(
        "--role", choices=sorted(ROLES), help="compare the role's scope only"
    )
    p.add_argument("--signals", help="narrow signals to these descriptors (comma list)")
    p.add_argument("--limit", type=int, default=20, help="paths listed at most")
    p = sub.add_parser(
        "sync", help="vendor a kotekan document as the golden, and pin it"
    )
    p.add_argument("--from", dest="source", required=True)
    p.add_argument("--commit", help="the kotekan commit, when the source cannot say")
    p.add_argument("--golden", default=GOLDEN_PATH, help=argparse.SUPPRESS)
    p.add_argument("--pin", default=PIN_PATH, help=argparse.SUPPRESS)
    p.add_argument("--limit", type=int, default=20, help="paths listed at most")
    p = sub.add_parser(
        "preflight",
        help="before a start: the binary's document against the golden and the config stamp",
    )
    p.add_argument("--role", required=True, choices=sorted(ROLES))
    p.add_argument("--config", help="the config the binary will start on")
    p.add_argument("--from", dest="source", default="-", help="default: stdin")
    p.add_argument("--signals", help="the descriptors to check (default: the config's)")
    p.add_argument("--timeout", type=float, default=3.0, help="for a URL source, s")
    p.add_argument("--limit", type=int, default=20, help="paths listed at most")
    p = sub.add_parser(
        "check-fleet",
        help="read every origin of a chains yaml once and print its verdict",
    )
    p.add_argument("chains", nargs="?", default=CHAINS_YAML)
    p.add_argument("--timeout", type=float, default=3.0, help="per read, s")
    a = ap.parse_args(argv)
    if a.cmd is None:
        ap.print_help()
        return 2
    try:
        return {
            "show": _cmd_show,
            "diff": _cmd_diff,
            "sync": _cmd_sync,
            "preflight": _cmd_preflight,
            "check-fleet": _cmd_check_fleet,
        }[a.cmd](a)
    except (SourceError, GoldenError) as e:
        print("iface %s: %s" % (a.cmd, e), file=sys.stderr)
        return 2
    except Exception:
        # 2: nothing was checked. Not 1, a flag day to diff and a MISMATCH to check-fleet.
        traceback.print_exc()
        print("iface %s: internal error, nothing was checked" % a.cmd, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
