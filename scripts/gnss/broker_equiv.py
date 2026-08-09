#!/usr/bin/env python3
"""The broker refactor's equivalence gate (task #27 M0, docs/CHORD_BROKER_REFACTOR.md).

`gps_distributed_broker.py` is 6400 lines with a 5100-line `main()`, and every comment block
in it is a scar from a real outage. Restructuring it on inspection is not a plan. So:

  1. RECORD a real run's entire conversation with the world
         broker_equiv.py record out.jsonl -- <the usual broker flags> --interval 2
     (or add `--transcript-write out.jsonl` to any existing launch script, which is what you
     want for the on-sky captures -- it is side-effect-free and costs a JSON line per poll.)

  2. REPLAY it against whatever the code has become, and hash the POST stream
         broker_equiv.py digest out.jsonl

  3. GATE a refactor: the digest must not move.
         broker_equiv.py check out.jsonl              # vs out.jsonl.digest
         broker_equiv.py bless out.jsonl              # record the current digest as golden

The POST stream is the right gate because it is the broker's entire output: seeds to the
trackers, search commands, nav bits, carrier trim. Two brokers that POST the same bytes in
the same order ARE the same broker as far as the instrument is concerned.

Byte-identical is a legitimate bar here (contrast the CUDA gates, where fmaf contraction
moves with kernel shape): Python neither contracts nor reassociates float arithmetic, so a
pure code move reproduces the numbers exactly. A moved digest is a real change.

  4. SELF-TEST -- because a gate that cannot fail is not a gate:
         broker_equiv.py selftest out.jsonl
     replays twice (must agree) and then replays with a one-part-in-1e12 perturbation
     injected into the loop gains (must DISAGREE). If the perturbed run still matches, the
     transcript does not exercise the code and the gate is decorative -- see the warning in
     docs/CHORD_BROKER_REFACTOR.md 3.2 about transcripts captured against a dead fleet.
"""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys

K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BROKER = os.path.join(K, "python", "scripts", "gnss", "gps_distributed_broker.py")
PY = os.environ.get("GNSS_PY", "/home/kvand/gnss/venv/bin/python")
# The pinned nav snapshot every replay runs against. Blob-side like the transcripts (NFS,
# not in git -- ~3.5 MB); the DIGEST records its fingerprint, so `git status` cannot see it
# drift but `check` can and says so.
BRDC_PIN = os.environ.get("GNSS_BRDC_PIN", "/home/kvand/gnss/fixtures/brdc_pin")


FIXTURES = os.path.join(K, "scripts", "gnss", "fixtures")


def _open(path):
    """Transcripts of a real fleet are ~1.3 MB per cycle, so they live gzipped and OUTSIDE
    the repo -- only the digest is versioned. Read either form transparently."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _gold_path(transcript):
    """THE GOLDEN DIGEST LIVES IN THE REPO, ALWAYS -- never beside the transcript.

    ⚠️ THIS WAS A REAL SILENT FAILURE (found 2026-08-08). The golden used to be
    `transcript + ".digest"`, which for the on-sky fixtures put it on NFS at
    /home/kvand/gnss/fixtures/ -- OUTSIDE git. `bless` wrote there, `check` read there, and
    the copy under scripts/gnss/fixtures/ was a hand-made mirror. The E5a mirror was
    committed carrying 776c70ff..., a number that NO commit in the history reproduces:
    blessed from a dirty tree, then the tree moved before the commit landed. The gate was
    green-or-red against a phantom for its whole life.

    A gate's expectation is source. Put it under version control and the failure cannot
    recur -- a stale golden becomes a diff in `git status` instead of a file nobody looks at.
    """
    return os.path.join(FIXTURES, os.path.basename(transcript) + ".digest")


def _tree_state():
    """(sha, dirty) of the tree that a digest is being blessed from.

    A digest is a claim about code. Recording WHICH code makes a stale golden diagnosable in
    one line instead of three worktree replays, and a dirty tree is recorded as dirty
    because a bless from uncommitted work describes something unreproducible.
    """
    def _git(*a):
        try:
            return subprocess.run(["git"] + list(a), cwd=K, capture_output=True,
                                  text=True).stdout.strip()
        except Exception:
            return ""
    sha = _git("rev-parse", "--short", "HEAD") or "unknown"
    # The harness counts too: `replay` builds the argv, so a change here can move the digest
    # every bit as much as a change in the broker.
    dirty = bool(_git("status", "--porcelain", "--", BROKER, os.path.abspath(__file__),
                      os.path.join(K, "python", "scripts", "gnss", "gnss_broker")))
    return sha, dirty


def _brdc_fingerprint():
    """Short hash of the BRDC nav files the replay will parse.

    ⚠️ THE GATE HAS A THIRD INPUT AND IT IS NOT THE TRANSCRIPT. A transcript freezes the
    fleet conversation, but the broadcast ephemeris is fetched by gnss_ephemeris.fetch_brdc
    through its OWN urllib -- not through transport.py -- into ~/.cache/kotekan_gps, and it
    is refreshed several times a day. So the digest is a function of (code, transcript,
    TODAY'S EPHEMERIS), and the third term changes on its own.

    Found the hard way 2026-08-09: the E5a golden was blessed at 20:36, the daily BRDC file
    was rewritten at 20:44, and every replay after that produced a different digest with
    IDENTICAL code -- one seed's cp0 moved 0.1 chips because a satellite's freshest toe had
    changed. I had already blamed the earlier instance of this on a dirty-tree bless, which
    was wrong: no tree was ever dirty, the sky's ephemeris simply moved on.

    A model-primary chain (E5a/B2a) is maximally exposed because EVERY seed comes from the
    model. GPS L5 is search-anchored and its digest survived the same update.

    Recording the fingerprint does not make the gate hermetic -- pinning the files would (see
    task #29) -- but it makes a moved digest self-explaining instead of a mystery that costs
    an afternoon.
    """
    cache = BRDC_PIN  # what replays READ -- never the live cache, which refreshes daily
    h = hashlib.sha256()
    try:
        for name in sorted(os.listdir(cache)):
            if not name.endswith((".rnx.gz", ".rnx")):
                continue
            p = os.path.join(cache, name)
            h.update(name.encode())
            h.update(str(os.path.getsize(p)).encode())
            with open(p, "rb") as f:
                h.update(f.read())
    except Exception:
        return "none"
    return h.hexdigest()[:12]


def _read_gold(path):
    """First token is the digest; the rest of the line is provenance. Old single-token
    goldens still read correctly."""
    with open(path) as f:
        head = f.readline().split()
    return (head[0] if head else ""), " ".join(head[1:])


def _header_argv(path):
    """The recording run's own flags, carried in the transcript's first line."""
    with _open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("k") == "argv":
                    return r["v"]
                break
    sys.exit("%s has no argv header -- recorded by a pre-M0 broker?" % path)


def _census(path):
    """What the transcript actually contains. A replay of nothing replays perfectly."""
    n = {"now": 0, "get": 0, "post": 0}
    urls, prns = set(), set()
    with _open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["k"] in n:
                n[r["k"]] += 1
            if r["k"] == "post":
                urls.add(r["u"])
    return n, urls, prns


def replay(path, extra=(), env=None):
    """Run the broker against a transcript; return (digest, stderr)."""
    argv = _header_argv(path)
    # --publish-port 0 disables the REST publisher: replays must not fight over a port, and
    # two of them running concurrently is the normal case when bisecting.
    argv = _strip(argv, "--publish-port")
    cmd = ([PY, "-u", BROKER] + argv
           + ["--transcript-read", path, "--publish-port", "0"] + list(extra))
    e = dict(os.environ)
    # PIN THE SKY (task #29). The ephemeris is a replay input every bit as much as the
    # transcript: gnss_ephemeris.fetch_brdc pulls it through its own urllib into a cache that
    # refreshes several times a day, and it moved the on-sky digests twice in two days with
    # byte-identical code -- the second time because midnight UTC rolled the day-of-year
    # mid-review. Replays read exactly the nav files in the pinned snapshot, so the digest is
    # a function of (code, transcript) and nothing else.
    e["GNSS_BRDC_DIR"] = BRDC_PIN
    e.update(env or {})
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=K)
    out = p.stdout.strip().splitlines()
    if not out:
        sys.stderr.write(p.stderr[-4000:] + "\n")
        sys.exit("replay produced no digest (broker exited %d)" % p.returncode)
    return out[-1].strip(), p.stderr


def _strip(argv, flag):
    keep, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == flag:
            skip = True
            continue
        if a.startswith(flag + "="):
            continue
        keep.append(a)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["record", "digest", "check", "bless", "selftest",
                                       "census"])
    ap.add_argument("transcript")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="for `record`: the broker flags to run under, after a bare --")
    a = ap.parse_args()
    gold = _gold_path(a.transcript)

    if a.action == "record":
        flags = [x for x in a.rest if x != "--"]
        if not flags:
            sys.exit("record needs the broker flags: broker_equiv.py record f.jsonl -- --trackers ...")
        cmd = [PY, "-u", BROKER] + flags + ["--transcript-write", a.transcript]
        sys.stderr.write("+ %s\n" % " ".join(cmd))
        raise SystemExit(subprocess.call(cmd, cwd=K))

    if a.action == "census":
        n, urls, _ = _census(a.transcript)
        print("cycles(now)=%d  gets=%d  posts=%d" % (n["now"], n["get"], n["post"]))
        for u in sorted(urls):
            print("  post ->", u)
        return

    if a.action == "digest":
        d, _ = replay(a.transcript)
        print(d)
        return

    if a.action == "bless":
        d, _ = replay(a.transcript)
        sha, dirty = _tree_state()
        prov = "blessed-at %s%s brdc %s" % (sha, " DIRTY" if dirty else "",
                                            _brdc_fingerprint())
        with open(gold, "w") as f:
            f.write("%s  %s\n" % (d, prov))
        print("blessed %s = %s (%s)" % (os.path.basename(gold), d, prov))
        if dirty:
            print("⚠️  BLESSED FROM A DIRTY TREE. This digest describes uncommitted broker\n"
                  "    code, so nothing in git reproduces it. Commit the broker change and\n"
                  "    re-bless, or the gate guards a phantom (that is exactly how the E5a\n"
                  "    golden was born).", file=sys.stderr)
        return

    if a.action == "check":
        if not os.path.exists(gold):
            sys.exit("no golden digest at %s -- run `bless` first" % gold)
        want, prov = _read_gold(gold)
        got, err = replay(a.transcript)
        if got == want:
            print("EQUIVALENT  %s%s" % (got, "  [%s]" % prov if prov else ""))
            return
        sys.stderr.write(err[-4000:] + "\n")
        now_brdc = _brdc_fingerprint()
        brdc_note = ""
        if prov and "brdc " in prov and ("brdc %s" % now_brdc) not in prov:
            brdc_note = (
                "\n\n⚠️  THE EPHEMERIS MOVED, NOT NECESSARILY THE CODE. This golden was "
                "blessed against\n    BRDC %s; the cache now hashes %s. The broadcast "
                "ephemeris is fetched\n    outside the transcript and refreshes several "
                "times a day, and a MODEL-PRIMARY chain\n    (E5a/B2a) seeds every satellite "
                "from it -- a changed toe moves a seed's cp0 with no\n    code change at "
                "all. Check that before concluding anything about the diff."
                % (prov.split("brdc ")[-1].split()[0], now_brdc))
        sys.exit("DIGEST MOVED\n  golden %s%s\n  now    %s\nThe POST stream changed: the "
                 "refactor is not behaviour-preserving.\nIf the change is intended, "
                 "`bless` FROM A CLEAN TREE and commit the .digest with the code.%s"
                 % (want, "  [%s]" % prov if prov else "", got, brdc_note))

    if a.action == "selftest":
        n, _, _ = _census(a.transcript)
        if n["post"] == 0:
            sys.exit("transcript contains ZERO posts -- it exercises nothing. Capture one "
                     "against a chain that is actually seeding (see CHORD_BROKER_REFACTOR "
                     "3.2: a frozen chain replays perfectly and tests nothing).")
        d1, _ = replay(a.transcript)
        d2, _ = replay(a.transcript)
        if d1 != d2:
            sys.exit("NOT DETERMINISTIC: two replays of the same transcript disagree\n"
                     "  %s\n  %s\nSomething outside the transcript is leaking in (a clock, "
                     "an unseeded RNG, a set iteration)." % (d1, d2))
        print("determinism OK   %s  (%d cycles, %d posts)" % (d1, n["now"], n["post"]))
        # The gate must be able to FAIL. Nudge each knob by one part in a million -- far
        # below anything physical (0.25 -> 0.25000025) -- and require the digest to move.
        #
        # SEVERAL KNOBS, NOT ONE, because a fixture only covers what it runs: the synthetic
        # fleet exercises the DLL, while the e2e fixture (real GPU, known truth) runs with
        # --dll-gain 0 and is purely a seed-arithmetic test. Perturbing only the loop gains
        # there reports GATE IS BLIND when the truth is "this fixture does not cover the
        # loops". So: report WHICH knobs move it -- that is a direct read on coverage.
        #
        # ⚠️ AND NOT 1e-12, which was the first attempt and cried blind on the synthetic
        # fixture too. That was not blindness, it was arithmetic: 1e-12 on a gain of 0.25
        # moves a ~0.01-chip trim by 1e-14, landing on a code phase of ~5000 chips where a
        # double resolves ~1e-12. The perturbation vanished BELOW the number it was added
        # to. A sensitivity test must clear the representation of what it perturbs.
        moved = []
        for knob in ("carrier_hz", "hops_per_sec", "dll_gain", "carrier_gain",
                     "code_bias_alpha", "bias_alpha"):
            d3, _ = replay(a.transcript,
                           env={"GNSS_BROKER_EQUIV_PERTURB": "%s:1e-6" % knob})
            if d3 != d1:
                moved.append(knob)
        if not moved:
            sys.exit("GATE IS BLIND: no knob moved the digest. The transcript's POSTs do "
                     "not depend on anything this broker computes -- it is inert, and this "
                     "gate would pass a broken refactor.")
        print("sensitivity OK   moved by: %s" % ", ".join(moved))
        skipped = [k for k in ("dll_gain", "carrier_gain", "code_bias_alpha", "bias_alpha")
                   if k not in moved]
        if skipped:
            print("coverage NOTE    this fixture does NOT reach: %s" % ", ".join(skipped))
        print("\nGATE GOOD.")
        return


if __name__ == "__main__":
    main()
