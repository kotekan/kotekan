#!/usr/bin/env python3
"""The GuoWang 1202.025 MHz satellite library, guowang_1202_db.json: read it, and fold the beam-dump
reductions (~/gnss/builds/gw1202/gw_reduce.py) into it.

The constellation is in a testing phase: some satellites transmit on 1202.025 MHz, some pass the
beam without a 1202 code, and some codes are shared. So the library keeps every satellite with a
conclusive pass, with its history:

- status: "transmitting" (its code was found on its newest conclusive pass) or "no_transmission"
  (no 1202 code on its newest conclusive pass). A pass is conclusive when it is a complete
  beam-dump window outside the no_sky periods and either the code was found (any separation) or
  the satellite passed within silent_max_sep_deg of the boresight, where every transmitting
  satellite is found. A miss farther out says nothing and is not counted.
- the code reference, once the code has been found: g2_phase, theta_us, carrier_minus_tle_ppb,
  measured_utc, source, sep_deg, ref_input, drift_ns_s, from the closest pass on which it was found
  (a closer pass replaces it only on the same reference input). g2_phase is null for a satellite
  never found. A satellite that stops keeps its code.
- last_observed_utc: the newest pass on which the code was found (null: never).
- last_checked_utc, last_checked_sep_deg: the newest conclusive pass.
- passes_found, passes_silent, and passes: every conclusive pass, [utc mid-window, closest
  separation in deg, found]. The list only grows, so the library survives the deletion of the
  reduce logs; the git history of the file is the time record.

no_sky lists periods whose windows carry no sky signal (analog chains off, ...): their windows
count for nothing. An open period (end null) excludes everything after its start.

    guowang_db.py update --reduce-log cf06_reduce_log.jsonl --reduce-log cf05_reduce_log.jsonl \\
        --proposals ~/gnss/builds/gw1202/db_proposals.jsonl [--dry-run] [--message-out msg.txt]
    guowang_db.py nosky --start 2026-11-03T16:34:50Z [--end 2026-11-04T01:00:00Z] --note "..."
    guowang_db.py show
"""
import argparse
import datetime as dt
import json
import os
import re
import sys

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "guowang_1202_db.json"
)
SCHEMA = "chord-gnss-guowang-1202-db/1"
SILENT_MAX_SEP_DEG = 4.0
CODE_FIELDS = (
    "g2_phase",
    "theta_us",
    "carrier_minus_tle_ppb",
    "measured_utc",
    "source",
    "sep_deg",
    "ref_input",
    "drift_ns_s",
    "note",
)
ABOUT_V1 = (
    " Schema 1 (2026-11-03): status is transmitting or no_transmission, from the newest conclusive"
    " pass: a complete beam-dump window outside the no_sky periods on which the code was found, or"
    " on which the satellite passed within silent_max_sep_deg of the boresight without it. g2_phase"
    " is null for a satellite never found; one that stops keeps its code. last_observed_utc is the"
    " newest pass with the code found, last_checked_utc the newest conclusive pass; passes lists"
    " every conclusive pass as [utc mid-window, closest separation deg, found]. guowang_db.py"
    " folds the reductions in; git history is the time record."
)


def utc(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s):
    s = s.rstrip("Z")
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in s else "%Y-%m-%dT%H:%M:%S"
    return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc)


def sat_key(name):
    m = re.search(r"(\d+)$", name)
    return (name.rsplit("-", 1)[0], int(m.group(1)) if m else 0, name)


def load(path=DB_PATH):
    with open(path) as fh:
        return json.load(fh)


def coded(db):
    """{name: entry} for the satellites whose code is known (g2_phase set), whatever their status."""
    return {n: s for n, s in db["satellites"].items() if s.get("g2_phase") is not None}


def one_line(m):
    return "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]"


def dumps(db):
    """indent-1 JSON, satellites in numeric order, every innermost list (a pass) on one line."""
    out = dict(db)
    out["satellites"] = {
        n: db["satellites"][n] for n in sorted(db["satellites"], key=sat_key)
    }
    text = json.dumps(out, indent=1, ensure_ascii=False)
    return re.sub(r"\[\s*([^\[\]{}]*?)\s*\]", one_line, text) + "\n"


def save(db, path=DB_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(dumps(db))
    os.replace(tmp, path)


def in_no_sky(db, t):
    for p in db.get("no_sky", []):
        if parse_utc(p["start"]) <= t and (
            p.get("end") is None or t < parse_utc(p["end"])
        ):
            return True
    return False


def window_mid(rec):
    """mid-window UTC of a reduce_log record (utc_start is a full stamp, utc_end a time of day)."""
    t0 = parse_utc(rec["utc_start"])
    h, m, s = rec["utc_end"].rstrip("Z").split(":")
    t1 = t0.replace(hour=int(h), minute=int(m), second=0, microsecond=0) + dt.timedelta(
        seconds=float(s)
    )
    if t1 < t0:
        t1 += dt.timedelta(days=1)
    return t0 + (t1 - t0) / 2


def read_passes(db, logs, silent_max_sep):
    """{sat: {utc: [utc, sep, found]}} of the conclusive passes in gw_reduce reduce_log.jsonl files,
    and the count of inconclusive ones (complete windows only, no_sky windows dropped)."""
    out, skipped = {}, {"incomplete": 0, "no_sky": 0, "far_miss": 0}
    for path in logs:
        try:
            fh = open(path)
        except OSError as exc:
            print("skip %s: %s" % (path, exc), file=sys.stderr)
            continue
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if not rec.get("complete") or rec.get("lobe_channels", 0) < 64:
                    skipped["incomplete"] += 1
                    continue
                t = window_mid(rec)
                if in_no_sky(db, t):
                    skipped["no_sky"] += 1
                    continue
                for s in rec.get("satellites", []):
                    sep = s.get("sep_min_window")
                    if sep is None and s.get("sep_range"):
                        sep = s["sep_range"][0]
                    found = bool(s.get("detected"))
                    if not found and (sep is None or sep > silent_max_sep):
                        skipped["far_miss"] += 1
                        continue
                    key = utc(t)
                    out.setdefault(s["sat"], {})[key] = [
                        key,
                        round(sep, 2) if sep is not None else None,
                        found,
                    ]
    return out, skipped


def read_proposals(paths):
    """{sat: [proposal, ...]} from gw_reduce db_proposals.jsonl files."""
    out = {}
    for path in paths:
        try:
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        p = json.loads(line)
                        out.setdefault(p["sat"], []).append(p)
        except OSError as exc:
            print("skip %s: %s" % (path, exc), file=sys.stderr)
    return out


def tle_catnums(path):
    """{'DIGUI-nn': NORAD number} from a TLE file (name lines 'HULIANWANG DIGUI-nn')."""
    out = {}
    if not path:
        return out
    lines = open(path).read().splitlines()
    for k in range(len(lines) - 2):
        if lines[k + 1].startswith("1 ") and lines[k + 2].startswith("2 "):
            out[lines[k].strip().replace("HULIANWANG ", "")] = int(lines[k + 1][2:7])
    return out


def newest_tle(cache=os.path.expanduser("~/.cache/guowang")):
    try:
        names = sorted(
            f
            for f in os.listdir(cache)
            if f.startswith("hulianwang_") and f.endswith(".tle")
        )
    except OSError:
        return None
    return os.path.join(cache, names[-1]) if names else None


def update(db, passes, proposals, catnums, silent_max_sep=SILENT_MAX_SEP_DEG):
    """Fold conclusive passes and code proposals into db (in place); returns the change list."""
    sats = db["satellites"]
    changes = []
    names = set(sats) | set(passes) | {n for n, ps in proposals.items() if ps}
    for name in sorted(names, key=sat_key):
        old = sats.get(name)
        e = dict(old) if old else {}
        # pass history: the stored list, the reduce logs, and the code reference's own pass
        hist = {p[0]: list(p) for p in e.get("passes", [])}
        if e.get("g2_phase") is not None and e.get("measured_utc") and not hist:
            hist[e["measured_utc"]] = [e["measured_utc"], e.get("sep_deg"), True]
        for key, p in passes.get(name, {}).items():
            near = [
                k
                for k in hist
                if abs((parse_utc(k) - parse_utc(key)).total_seconds()) < 300
            ]
            if (
                near
            ):  # the same pass (the code reference's closest-approach stamp, or a re-reduction)
                for k in near:
                    del hist[k]
            hist[key] = p
        # code reference: the closest pass on which the code was found, on the same reference input
        ref = {k: e[k] for k in CODE_FIELDS if k in e and e.get("g2_phase") is not None}
        for p in proposals.get(name, []):
            if ref and p["g2_phase"] != ref["g2_phase"]:
                changes.append(
                    "%s: CODE CONFLICT, pass %s has G2' %d, library %d (library kept)"
                    % (name, p["measured_utc"], p["g2_phase"], ref["g2_phase"])
                )
                continue
            if ref and p.get("ref_input", ref.get("ref_input")) != ref.get("ref_input"):
                continue
            if not ref or p["sep_deg"] < ref["sep_deg"] - 1e-9:
                if ref:
                    changes.append(
                        "%s: code reference now the %.2f deg pass of %s (was %.2f deg)"
                        % (name, p["sep_deg"], p["measured_utc"], ref["sep_deg"])
                    )
                ref = {k: p[k] for k in CODE_FIELDS if k in p}
            if not any(
                abs((parse_utc(k) - parse_utc(p["measured_utc"])).total_seconds()) < 300
                for k in hist
            ):
                hist[p["measured_utc"]] = [p["measured_utc"], p["sep_deg"], True]
        if not hist and not ref:
            continue
        plist = sorted(hist.values(), key=lambda p: p[0])
        found = [p for p in plist if p[2]]
        newest = plist[-1] if plist else None
        n = {
            "status": ("transmitting" if newest[2] else "no_transmission")
            if newest
            else "transmitting"
        }
        n["catnum"] = (
            e.get("catnum")
            or next(
                (p.get("catnum") for p in proposals.get(name, []) if p.get("catnum")),
                None,
            )
            or catnums.get(name)
        )
        if ref:
            n.update(ref)
        else:
            n["g2_phase"] = None
        n["last_observed_utc"] = found[-1][0] if found else None
        n["last_checked_utc"] = newest[0] if newest else None
        n["last_checked_sep_deg"] = newest[1] if newest else None
        n["passes_found"] = len(found)
        n["passes_silent"] = len(plist) - len(found)
        n["passes"] = plist
        if old is None:
            changes.append(
                "%s: NEW, %s%s (newest pass %s, %.2f deg)"
                % (
                    name,
                    n["status"],
                    ", G2' %d" % n["g2_phase"] if n["g2_phase"] is not None else "",
                    n["last_checked_utc"],
                    n["last_checked_sep_deg"]
                    if n["last_checked_sep_deg"] is not None
                    else float("nan"),
                )
            )
        else:
            if old.get("status", "transmitting") != n["status"]:
                changes.append(
                    "%s: %s -> %s on the %s pass (%.2f deg)"
                    % (
                        name,
                        old.get("status", "transmitting"),
                        n["status"],
                        n["last_checked_utc"],
                        n["last_checked_sep_deg"],
                    )
                )
            if old.get("g2_phase") is None and n["g2_phase"] is not None:
                changes.append("%s: code found, G2' %d" % (name, n["g2_phase"]))
        sats[name] = n
    db["schema"] = SCHEMA
    db["silent_max_sep_deg"] = silent_max_sep
    db.setdefault("no_sky", [])
    if ABOUT_V1.strip() not in db.get("about", ""):
        db["about"] = db.get("about", "") + ABOUT_V1
    return changes


def summary_line(db):
    s = db["satellites"].values()
    tx = sum(1 for e in s if e["status"] == "transmitting")
    nt = sum(1 for e in s if e["status"] == "no_transmission")
    codes = {}
    for e in s:
        if e.get("g2_phase") is not None:
            codes.setdefault(e["g2_phase"], 0)
            codes[e["g2_phase"]] += 1
    shared = sum(1 for v in codes.values() if v > 1)
    return (
        "%d satellites: %d transmitting, %d no_transmission; %d codes (%d shared)"
        % (len(db["satellites"]), tx, nt, len(codes), shared,)
    )


def cmd_update(a):
    db = load(a.db)
    passes, skipped = read_passes(db, a.reduce_log, a.silent_max_sep)
    proposals = read_proposals(a.proposals)
    changes = update(
        db, passes, proposals, tle_catnums(a.tle or newest_tle()), a.silent_max_sep
    )
    before = open(a.db).read()
    text = dumps(db)
    changed = text != before
    print(summary_line(db))
    print("skipped windows/passes: %s" % skipped)
    for c in changes:
        print("  " + c)
    if a.message_out:
        new = sum(1 for c in changes if ": NEW," in c)
        flips = sum(1 for c in changes if " -> " in c)
        title = "gnss: GuoWang 1202 library: %s" % summary_line(db)
        if new or flips:
            title = "gnss: GuoWang 1202 library: %d new, %d status changes" % (
                new,
                flips,
            )
        body = [summary_line(db) + "."]
        if changes:
            body.append("")
            body += ["- " + c for c in changes]
        else:
            body.append("")
            body.append("Newer passes only (last_observed / last_checked).")
        body += [
            "",
            "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
            "Claude-Session: https://claude.ai/code/session_0161577TDGeS2hdLuBmmSPhD",
        ]
        with open(a.message_out, "w") as fh:
            fh.write(title + "\n\n" + "\n".join(body) + "\n")
    if a.dry_run:
        print("dry run: %s" % ("would change the library" if changed else "no change"))
        return
    if changed:
        save(db, a.db)
        print("wrote %s" % a.db)
    else:
        print("no change")


def cmd_nosky(a):
    db = load(a.db)
    periods = db.setdefault("no_sky", [])
    for p in periods:
        if p["start"] == a.start:
            if a.end:
                p["end"] = a.end
            if a.note:
                p["note"] = a.note
            break
    else:
        periods.append(dict(start=a.start, end=a.end, note=a.note or ""))
    periods.sort(key=lambda p: p["start"])
    save(db, a.db)
    print(json.dumps(periods, indent=1))


def cmd_show(a):
    db = load(a.db)
    print(summary_line(db))
    print(
        "%-11s %-15s %5s %-20s %-20s %6s %3s %3s"
        % ("sat", "status", "G2'", "last_observed", "last_checked", "sep", "+", "-")
    )
    for n in sorted(db["satellites"], key=sat_key):
        e = db["satellites"][n]
        print(
            "%-11s %-15s %5s %-20s %-20s %6s %3d %3d"
            % (
                n,
                e.get("status", "transmitting"),
                e.get("g2_phase") if e.get("g2_phase") is not None else "-",
                e.get("last_observed_utc") or "-",
                e.get("last_checked_utc") or "-",
                e.get("last_checked_sep_deg")
                if e.get("last_checked_sep_deg") is not None
                else "-",
                e.get("passes_found", 0),
                e.get("passes_silent", 0),
            )
        )
    for p in db.get("no_sky", []):
        print(
            "no_sky %s .. %s: %s"
            % (p["start"], p.get("end") or "open", p.get("note", ""))
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", default=DB_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser(
        "update", help="fold reduce logs and code proposals into the library"
    )
    u.add_argument(
        "--reduce-log",
        action="append",
        default=[],
        help="gw_reduce reduce_log.jsonl (repeatable)",
    )
    u.add_argument(
        "--proposals",
        action="append",
        default=[],
        help="gw_reduce db_proposals.jsonl (repeatable)",
    )
    u.add_argument(
        "--tle",
        help="TLE file for NORAD numbers (default: the newest ~/.cache/guowang file)",
    )
    u.add_argument("--silent-max-sep", type=float, default=SILENT_MAX_SEP_DEG)
    u.add_argument("--message-out", help="write a commit message here")
    u.add_argument("--dry-run", action="store_true")
    n = sub.add_parser(
        "nosky", help="add or close a no-sky period (windows inside count for nothing)"
    )
    n.add_argument("--start", required=True, help="UTC, e.g. 2026-11-03T16:34:50Z")
    n.add_argument("--end", help="UTC; omit for an open period")
    n.add_argument("--note")
    s = sub.add_parser("show", help="one line per satellite")
    u.set_defaults(func=cmd_update)
    n.set_defaults(func=cmd_nosky)
    s.set_defaults(func=cmd_show)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
