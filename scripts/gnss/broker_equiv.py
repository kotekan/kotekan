#!/usr/bin/env python3
"""Equivalence gate for broker refactors: record a broker run, replay it, hash the POST stream.

  1. RECORD a real run's entire conversation with the world
         broker_equiv.py record out.jsonl -- <the usual broker flags> --interval 2
     (or add `--transcript-write out.jsonl` to an existing launch; it is side-effect-free and
     costs a JSON line per poll).

  2. REPLAY it against the current code and hash the POST stream
         broker_equiv.py digest out.jsonl

  3. GATE a refactor: the digest must not move.
         broker_equiv.py check out.jsonl    # vs scripts/gnss/fixtures/out.jsonl.digest
         broker_equiv.py bless out.jsonl    # record the current digest as golden

     A replay runs under the transcript's own flags, so a knob set only in the config and in
     no recording (--cp-rate-model-tol / --fit-flush-on-reject) is code the plain golden
     never executes. A variant golden pins the same recording with flags appended to the
     transcript's, under its own name:
         broker_equiv.py digest out.jsonl -- --cp-rate-model-tol 0.5 --fit-flush-on-reject 3
         broker_equiv.py bless --variant cprate out.jsonl -- --cp-rate-model-tol 0.5 ...
     The golden is out.jsonl.cprate.digest and records the appended flags beside the
     digest; `check out.jsonl` verifies every golden of the transcript, each under its
     own flags.

The POST stream is the broker's entire output: seeds to the trackers, search commands, nav
bits, carrier trim. Two brokers that POST the same bytes in the same order are the same
broker as far as the instrument is concerned.

Byte-identical is a fair bar here (unlike the CUDA gates, where fmaf contraction moves with
kernel shape): Python neither contracts nor reassociates float arithmetic, so a pure code
move reproduces the numbers exactly. A moved digest is a real change.

  4. SELF-TEST, because a gate that cannot fail is not a gate:
         broker_equiv.py selftest out.jsonl
     refuses a transcript with no POSTs, replays twice (must agree), then replays once per
     knob (carrier_hz, hops_per_sec, dll_gain, code_bias_alpha, bias_alpha) scaled by
     1 + 1e-6. At least one must move the digest, or the transcript does not exercise the
     code and the gate is decorative; the knobs that do move it are the fixture's coverage.

Goldens and the fixture list are in scripts/gnss/fixtures/ (README.md there).
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import shlex
import subprocess
import sys

K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BROKER = os.path.join(K, "python", "scripts", "gnss", "gps_distributed_broker.py")
PY = os.environ.get("GNSS_PY", "/home/kvand/gnss/venv/bin/python")
# The shared pinned nav snapshot replays read when a transcript has no sidecar (NFS, not in
# git). A golden records its fingerprint, so a `check` whose digest moved says whether the pin
# changed too.
BRDC_PIN = os.environ.get("GNSS_BRDC_PIN", "/home/kvand/gnss/fixtures/brdc_pin")


def _pin_for(transcript):
    """The BRDC snapshot this transcript replays against: `<transcript>.brdc/` if present,
    else the shared pin.

    The shared pin covers only the days its fixtures were captured on. Adding a new day's
    nav files to it would change what every fixture without a sidecar parses, and so the pin's
    fingerprint and possibly those fixtures' digests; a new capture carries its own snapshot as
    a sidecar instead.
    """
    side = transcript + ".brdc"
    return side if os.path.isdir(side) else BRDC_PIN


FIXTURES = os.path.join(K, "scripts", "gnss", "fixtures")

# Broker flags retired after some fixtures were recorded, with the number of values each takes.
# A transcript carries its argv verbatim, so replay drops these first; an unchanged digest is
# what shows the retired flag did nothing on that recording.
RETIRED_FLAGS = {
    "--carrier-gain": 1,
    "--carrier-from-code": 0,
    "--cl-assist": 0,
    "--eph-rebase": 1,
    "--fleet-trim-rebase-adjust": 1,
    "--innov-dr-seeds": 1,
    "--joint-feed-spec": 0,
    "--joint-model-primary": 0,
    "--joint-rereference": 0,
    "--joint-shadow": 0,
    "--nh-hint-span": 1,
    "--nh-period-offset": 1,
    "--spectrum-aligned": 0,
    "--spectrum-archive": 1,
    "--state-dongle": 1,
    "--state-file": 1,
    "--state-read-dir": 1,
}


def _open(path):
    """Open a transcript, gzipped (`.gz`) or plain. On-sky transcripts are large and live
    gzipped outside the repo; only their digests are versioned."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _gold_path(transcript, variant=None):
    """The golden's path, scripts/gnss/fixtures/<transcript name>[.<variant>].digest: in the
    repo, never beside the transcript.

    A gate's expectation is source. On-sky transcripts live on NFS outside git, so a golden
    beside them would be unversioned; in the repo a stale golden shows up in `git status`.
    """
    name = os.path.basename(transcript) + (".%s" % variant if variant else "")
    return os.path.join(FIXTURES, name + ".digest")


def _goldens(transcript):
    """The plain golden first, then every variant (`<transcript>.<name>.digest`)."""
    return [_gold_path(transcript)] + sorted(glob.glob(_gold_path(transcript, "*")))


def _tree_state():
    """(sha, dirty) of the tree that a digest is being blessed from.

    A digest is a claim about code. Recording which commit makes a stale golden diagnosable.
    Uncommitted changes to the broker, gnss_broker/ or this harness are flagged dirty, because
    a bless from uncommitted work cannot be reproduced; other modules the broker imports are
    not checked.
    """

    def _git(*a):
        try:
            return subprocess.run(
                ["git"] + list(a), cwd=K, capture_output=True, text=True
            ).stdout.strip()
        except Exception:
            return ""

    sha = _git("rev-parse", "--short", "HEAD") or "unknown"
    # The harness counts too: `replay` builds the argv, so a change here can move the digest
    # as much as a change in the broker.
    dirty = bool(
        _git(
            "status",
            "--porcelain",
            "--",
            BROKER,
            os.path.abspath(__file__),
            os.path.join(K, "python", "scripts", "gnss", "gnss_broker"),
        )
    )
    return sha, dirty


def _brdc_fingerprint(cache=None):
    """Short hash of the .rnx/.rnx.gz nav files in `cache` (default: the shared pin), "none"
    if it cannot be read.

    The ephemeris is a replay input the transcript does not hold: the broker fetches it
    through gnss_brdc_supply's own urllib, not through transport.py. Replays read a pinned
    snapshot (GNSS_BRDC_DIR), and `bless` records this fingerprint beside the digest so that
    a digest moved by a changed snapshot says so. A model-primary chain (E5a/B2a) is the most
    exposed, since every seed comes from the model.
    """
    cache = (
        cache or BRDC_PIN
    )  # what replays read, never the live cache, which refreshes several times a day
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
    """(digest, provenance): the first token of the golden's first line, then the rest of
    that line ("" for a golden that holds only the digest)."""
    with open(path) as f:
        head = f.readline().split()
    return (head[0] if head else ""), " ".join(head[1:])


def _gold_argv(prov):
    """The flags a variant golden was blessed under, appended to the transcript's own."""
    return shlex.split(prov.split(" argv ", 1)[1]) if " argv " in prov else []


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
    with _open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["k"] in n:
                n[r["k"]] += 1
    return n


def replay(path, env=None, extra=()):
    """Run the broker against a transcript; return (digest, stderr)."""
    pin = _pin_for(path)
    argv = _header_argv(path)
    # --publish-port 0 disables the REST publisher: replays must not fight over a port, and
    # two of them running concurrently is the normal case when bisecting.
    argv = _strip(argv, "--publish-port")
    for flag, nargs in RETIRED_FLAGS.items():
        argv = _strip(argv, flag, nargs)
    # `extra` goes after the transcript's flags: argparse keeps the last value, so a
    # golden's flags override the recording's.
    cmd = [PY, "-u", BROKER] + argv + list(extra)
    cmd += ["--transcript-read", path, "--publish-port", "0"]
    e = dict(os.environ)
    # Pin the sky. The ephemeris is a replay input as much as the transcript, and the live
    # cache refreshes several times a day (and rolls with the UTC day-of-year). Replays read
    # exactly the nav files in the pinned snapshot, so the digest is a function of
    # (code, transcript) and nothing else.
    e["GNSS_BRDC_DIR"] = pin or BRDC_PIN
    e.update(env or {})
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=K)
    out = p.stdout.strip().splitlines()
    if not out:
        sys.stderr.write(p.stderr[-4000:] + "\n")
        sys.exit("replay produced no digest (broker exited %d)" % p.returncode)
    return out[-1].strip(), p.stderr


def _strip(argv, flag, nargs=1):
    keep, skip = [], 0
    for a in argv:
        if skip:
            skip -= 1
            continue
        if a == flag:
            skip = nargs
            continue
        if a.startswith(flag + "="):
            continue
        keep.append(a)
    return keep


def _check(transcript, gold):
    """One golden against one replay; exits on a moved digest."""
    if not os.path.exists(gold):
        sys.exit("no golden digest at %s -- run `bless` first" % gold)
    want, prov = _read_gold(gold)
    got, err = replay(transcript, extra=_gold_argv(prov))
    if got == want:
        print(
            "EQUIVALENT  %s  %s%s"
            % (os.path.basename(gold), got, "  [%s]" % prov if prov else "")
        )
        return
    sys.stderr.write(err[-4000:] + "\n")
    now_brdc = _brdc_fingerprint(_pin_for(transcript))
    brdc_note = ""
    if prov and "brdc " in prov and ("brdc %s" % now_brdc) not in prov:
        brdc_note = (
            "\n\n⚠️  THE EPHEMERIS MOVED, NOT NECESSARILY THE CODE. This golden was "
            "blessed against\n    BRDC %s; the cache now hashes %s. The broadcast "
            "ephemeris is fetched\n    outside the transcript and refreshes several "
            "times a day, and a MODEL-PRIMARY chain\n    (E5a/B2a) seeds every satellite "
            "from it -- a changed toe moves a seed's cp0 with no\n    code change at "
            "all. Check that before concluding anything about the diff."
            % (prov.split("brdc ")[-1].split()[0], now_brdc)
        )
    sys.exit(
        "DIGEST MOVED  %s\n  golden %s%s\n  now    %s\nThe POST stream changed: the "
        "refactor is not behaviour-preserving.\nIf the change is intended, "
        "`bless` FROM A CLEAN TREE and commit the .digest with the code.%s"
        % (
            os.path.basename(gold),
            want,
            "  [%s]" % prov if prov else "",
            got,
            brdc_note,
        )
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "action", choices=["record", "digest", "check", "bless", "selftest"]
    )
    ap.add_argument("transcript")
    ap.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="for `record`: the broker flags to run under, after a bare --; for "
        "`digest`/`bless`: flags appended to the transcript's own",
    )
    ap.add_argument(
        "--variant",
        help="for `bless`, given BEFORE the transcript (what follows it is the replay's "
        "argv): write `<transcript>.<variant>.digest`, recording the flags "
        "after -- in it; for `check`: verify only that golden",
    )
    a = ap.parse_args()
    extra = [x for x in a.rest if x != "--"]
    # Absolute before anything uses it: the replay runs the broker with cwd=K (the repo root),
    # so a path relative to the caller's directory (`check fixtures/broker_fake_l5.jsonl` from
    # scripts/gnss) would resolve against the wrong directory in the child.
    a.transcript = os.path.abspath(a.transcript)
    gold = _gold_path(a.transcript, a.variant)

    if a.action == "record":
        if not extra:
            sys.exit(
                "record needs the broker flags: broker_equiv.py record f.jsonl -- --trackers ..."
            )
        cmd = [PY, "-u", BROKER] + extra + ["--transcript-write", a.transcript]
        sys.stderr.write("+ %s\n" % " ".join(cmd))
        raise SystemExit(subprocess.call(cmd, cwd=K))

    if a.action == "digest":
        d, _ = replay(a.transcript, extra=extra)
        print(d)
        return

    if a.action == "bless":
        if bool(a.variant) != bool(extra):
            sys.exit(
                "a variant golden is the transcript's flags PLUS the ones after --: "
                "broker_equiv.py bless --variant <name> f.jsonl -- <flags>"
            )
        d, _ = replay(a.transcript, extra=extra)
        sha, dirty = _tree_state()
        prov = "blessed-at %s%s brdc %s" % (
            sha,
            " DIRTY" if dirty else "",
            _brdc_fingerprint(_pin_for(a.transcript)),
        )
        if extra:
            prov += " argv " + shlex.join(extra)
        with open(gold, "w") as f:
            f.write("%s  %s\n" % (d, prov))
        print("blessed %s = %s (%s)" % (os.path.basename(gold), d, prov))
        if dirty:
            print(
                "⚠️  BLESSED FROM A DIRTY TREE. This digest describes uncommitted broker\n"
                "    code, so nothing in git reproduces it. Commit the broker change and\n"
                "    re-bless, or the gate guards a phantom (that is exactly how the E5a\n"
                "    golden was born).",
                file=sys.stderr,
            )
        return

    if a.action == "check":
        if extra:
            sys.exit("check takes a golden's flags from the golden itself")
        for g in [gold] if a.variant else _goldens(a.transcript):
            _check(a.transcript, g)
        return

    if a.action == "selftest":
        n = _census(a.transcript)
        if n["post"] == 0:
            sys.exit(
                "transcript contains ZERO posts -- it exercises nothing. Capture one "
                "against a chain that is actually seeding (see 31896a862:docs/CHORD_BROKER_REFACTOR.md "
                "3.2: a frozen chain replays perfectly and tests nothing)."
            )
        d1, _ = replay(a.transcript)
        d2, _ = replay(a.transcript)
        if d1 != d2:
            sys.exit(
                "NOT DETERMINISTIC: two replays of the same transcript disagree\n"
                "  %s\n  %s\nSomething outside the transcript is leaking in (a clock, "
                "an unseeded RNG, a set iteration)." % (d1, d2)
            )
        print("determinism OK   %s  (%d cycles, %d posts)" % (d1, n["now"], n["post"]))
        # The gate must be able to fail. Scale each knob by 1 + 1e-6, far below anything
        # physical (0.25 -> 0.25000025), and require the digest to move.
        #
        # Several knobs, because a fixture covers only what it runs: a model-primary chain
        # with no detections, or one run with --dll-gain 0, never reaches the DLL, and
        # perturbing only the loop gains would call it blind. The knobs that move the digest
        # are a direct read on coverage.
        #
        # 1e-6 and not smaller: the perturbation must survive the arithmetic it feeds. 1e-12
        # on a gain of 0.25 moves a ~0.01-chip trim by ~1e-14, below the ~1e-12 resolution of
        # a double at a code phase of ~5000 chips.
        moved = []
        for knob in (
            "carrier_hz",
            "hops_per_sec",
            "dll_gain",
            "code_bias_alpha",
            "bias_alpha",
        ):
            d3, _ = replay(
                a.transcript, env={"GNSS_BROKER_EQUIV_PERTURB": "%s:1e-6" % knob}
            )
            if d3 != d1:
                moved.append(knob)
        if not moved:
            sys.exit(
                "GATE IS BLIND: no knob moved the digest. The transcript's POSTs do "
                "not depend on anything this broker computes -- it is inert, and this "
                "gate would pass a broken refactor."
            )
        print("sensitivity OK   moved by: %s" % ", ".join(moved))
        skipped = [
            k for k in ("dll_gain", "code_bias_alpha", "bias_alpha") if k not in moved
        ]
        if skipped:
            print(
                "coverage NOTE    this fixture does NOT reach: %s" % ", ".join(skipped)
            )
        print("\nGATE GOOD.")
        return


if __name__ == "__main__":
    main()
