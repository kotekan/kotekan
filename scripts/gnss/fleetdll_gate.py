#!/usr/bin/env python3
"""F1 GATE (task #51): does the C++ fleet discriminator agree with the Python one?

Builds a fixture of telemetry frames, hands the SAME BYTES to both arms, and compares the
discriminator they produce for every PRN:

    Python:  gnss_broker.telem.TelemFrame -> TelemClient._store_frame -> combdll.fleet_dll_comb
    C++:     scripts/gnss/fleetdll  ->  gnss::FleetDll  (lib/stages/gnss/gnssFleetDll.hpp)

Neither arm is a model of the other: both are the shipped code. The Python side goes through
the REAL client ring (so `windows(lag=1)` and the eviction policy are exercised, not
reimplemented), and the C++ side is the same header GnssFleetTrim runs in production.

WHY BYTES AND NOT NUMBERS. Every transport defect this project has paid for lived in a
CONVENTION between two components -- what a number means, what it is reduced modulo, which
epoch it references -- and none was visible to a test of either component alone. A fixture of
floats handed to two parsers tests the parsers. A fixture of BYTES tests the contract.

    ./fleetdll_gate.py                 # build, run both arms, compare
    ./fleetdll_gate.py --self-test     # ALSO prove the comparison can fail (see below)
    ./fleetdll_gate.py --keep DIR      # leave the fixture and both answers behind

⚠️ THE GATE MUST BE ABLE TO FAIL. --self-test perturbs one channel of one record by 1% and
requires the comparison to report a difference. A gate that only ever passes has told you
nothing, and this repository has shipped two of those (docs/CHORD_FAST_TRIM.md 7b).

⚠️ AND IT MUST BE PRODUCTION-SIZED. fast_trim_e2e.py runs a 3-PRN fixture and could not catch
the regression where 5 Hz was requested and 1.5 Hz delivered -- it reached 4.39 Hz either way.
This one carries PRN and instance counts in the production range for the same reason, even
though F1 measures no rate: the shapes a bug hides in (row compaction, an instance with a
different channel count, a PRN below min_instances) do not appear at three PRNs.
"""
import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "python", "scripts", "gnss"))

from gnss_broker import combdll, telem  # noqa: E402


# ---------------------------------------------------------------------------------------------
# The fixture. Deliberately built from the SAME struct the broker parses with (telem._HDR), so a
# layout drift fails here as a struct error rather than as quietly wrong numbers.
# ---------------------------------------------------------------------------------------------

N_REC = 4                 # records per frame -- production
N_PRN = 12                # rows per record   -- production is 16 after #64
HOPS_PER_RECORD = 2048
FFT_LEN = 16384
SPACING = 0.5             # E/L spacing, chips

#: (instance, n_chan). Instances carry DIFFERENT channel counts on purpose: n_chan is a
#: per-sender header field, and a consumer that assumes one value reads the next row's floats.
INSTANCES = [("cx19.0", 7), ("cx19.1", 6), ("cx27.0", 7), ("cx27.1", 8),
             ("cx42.0", 7), ("cx42.1", 6)]

#: prn -> (code offset in chips, amplitude). Offset 0 = on peak.
#: Spread across the pull-in region so `disc` is not a single value: a gate whose fixture
#: produces one number cannot tell a working discriminator from a constant.
TARGETS = {4: (0.00, 1.00),    # on peak: disc ~ 0, q high
           9: (0.22, 0.80),    # early shoulder
           27: (-0.31, 0.65),  # late shoulder
           3: (0.55, 0.40),    # far out
           11: (0.10, 0.50),   # ONE instance only -> must be excluded by min_instances
           16: (0.00, 0.02),   # essentially noise
           # ⚠️ THE SIGNAL-FREE POPULATION IS LOAD-BEARING, NOT PADDING (added 2026-08-23).
           # The C++ integrator's information test is `p_pow >= sig_k * MEDIAN(p_pow over this
           # window's rows)` -- self-calibrating on the assumption that most PRNs are
           # signal-free at any moment, which is true on sky (24-31 rows per chain, a handful
           # live) and was NOT true of this fixture. With only 4/9/27/3/16 producing rows the
           # median WAS a signal level: floor 1.156e-01 against a strongest target of 8.78e-02,
           # so every armed PRN was SKIPPED, n_steps stayed 0, and the F2 leg could not run.
           # It reported "the C++ integrator stepped on nothing" -- which read as a code fault
           # and was a fixture fault. Same disease as the broker's own presence bar before
           # --noise-probes: a median over a population with no signal-free members is not a
           # noise level. These six carry rows and no signal, so the floor calibrates.
           5: (0.00, 0.02), 12: (0.00, 0.015), 18: (0.00, 0.025),
           21: (0.00, 0.018), 24: (0.00, 0.022), 30: (0.00, 0.017)}

#: PRNs the F2 leg arms: real signal only. 11 is the min_instances exclusion; the noise PRNs
#: above are deliberately NOT armed -- they exist to calibrate the floor, and arming them would
#: assert that the loop steps on noise, which is the one thing the floor exists to prevent.
SIGNAL_PRNS = [3, 4, 9, 27]


def _R(x):
    """Triangular code correlation, chips. The fixture's only physics."""
    return max(0.0, 1.0 - abs(x))


def _chan_block(rng, prn, fid, dead_energy, dead_el_energy):
    """One channel's nine floats: E, P, L as the assembler writes them (raw, un-normalised)."""
    d, amp = TARGETS[prn]
    if dead_energy:
        # No live comb this channel this record -- the assembler leaves the energy at zero and
        # the consumer must SKIP it. Zeroing it in instead would dilute the power exactly the
        # way the deep fold's zero-padding did.
        return [0.0] * 9
    e_p = 40.0 + 3.0 * (fid % 5)          # replica energy: per-channel, not per-tap
    phi = 0.7 * fid + 0.3 * prn           # a per-channel phase, so |sum| != sum|.|
    noise = lambda: rng.gauss(0.0, 0.02 * amp + 0.01)

    def tap(off, energy):
        a = amp * _R(d + off) + noise()
        return [a * energy * math.cos(phi), a * energy * math.sin(phi), energy]

    e_e = 0.0 if dead_el_energy else e_p * 0.98
    e_l = 0.0 if dead_el_energy else e_p * 1.02
    # ⚠️ When an E/L energy is exactly 0 both arms must fall back to the PROMPT energy
    # (Python: `a[CHAN_E_ENERGY] or eP`). The numerator is the raw complex either way, so this
    # is only visible in the denominator -- which is precisely the kind of difference that
    # would survive a casual comparison.
    p = tap(0.0, e_p)
    e = tap(+SPACING, e_e if e_e else e_p)
    l = tap(-SPACING, e_l if e_l else e_p)
    return [p[0], p[1], p[2], e[0], e[1], e_e, l[0], l[1], e_l]


def build_frame(chain, inst, n_chan, win, seq, prn_rows, present, rng, dead):
    """One wire frame, bytes. `prn_rows` is the row->PRN map (row compaction, #64)."""
    row_total = telem._ROW_FLOATS + telem._MAX_CHAN * telem._CHAN_FLOATS
    wstart0 = win * N_REC * HOPS_PER_RECORD * FFT_LEN
    chan_ids = [100 + 4 * i for i in range(n_chan)] + [0] * (telem._MAX_CHAN - n_chan)
    hdr = telem._HDR.pack(
        telem._MAGIC, telem._VERSION, N_REC, N_PRN, telem._ROW_FLOATS, n_chan, 32,
        HOPS_PER_RECORD, FFT_LEN, win, seq, wstart0, 0.0, present,
        telem._MAX_CHAN, row_total, chain.encode(), inst.encode(), *chan_ids)

    rows = [0.0] * (N_REC * N_PRN * row_total)
    for r in range(N_REC):
        for p, prn in enumerate(prn_rows):
            base = (r * N_PRN + p) * row_total
            # PRN is written on EVERY row of EVERY record, including records that did not run
            # -- that is what lets the row map be read from the data instead of configured.
            rows[base + telem.REC_PRN] = float(prn)
            if prn == 0 or not (present & (1 << r)):
                continue
            for ch in range(n_chan):
                cb = base + telem._ROW_FLOATS + ch * telem._CHAN_FLOATS
                blk = _chan_block(rng, prn, chan_ids[ch],
                                  dead_energy=(ch, prn) in dead["energy"],
                                  dead_el_energy=(ch, prn) in dead["el_energy"])
                rows[cb:cb + telem._CHAN_FLOATS] = blk
    return hdr + struct.pack("<%df" % len(rows), *rows)


def build_fixture(n_win, seed=20260815):
    """[(bytes, chain)] in DELIVERY ORDER -- window-major, as the live path sees them."""
    rng = random.Random(seed)
    dead = {"energy": {(2, 9), (5, 27)},      # two channels with no live comb
            "el_energy": {(0, 4)}}            # one channel predating the E/L energies
    frames = []
    for win in range(n_win):
        for chain in ("gps_l5", "gal_e5a"):
            for i, (inst, n_chan) in enumerate(INSTANCES):
                # ROW COMPACTION (#64): the row->PRN map differs per instance and is NOT the
                # configured PRN order. An arm that assumed row == PRN index would read another
                # satellite's comb and produce a confident wrong discriminator.
                prns = [4, 9, 27, 3, 16, 5, 12, 18, 21, 24, 30]
                if i == 0:
                    prns = prns + [11]        # PRN 11 on ONE instance only
                order = list(prns)
                rng.shuffle(order)
                prn_rows = order + [0] * (N_PRN - len(order))
                # A record slot that did not run is a HOLE AT A KNOWN INDEX. Vary which.
                present = 0xF if (win + i) % 4 else 0xD
                frames.append((build_frame(chain, inst, n_chan, win, win * 100 + i,
                                           prn_rows, present, rng, dead), chain))
    return frames


# ---------------------------------------------------------------------------------------------
# The two arms
# ---------------------------------------------------------------------------------------------

def write_stream(path, frames):
    """The gather's own wire protocol: [uint32 LE length][length bytes], repeated."""
    with open(path, "wb") as fh:
        for buf, _chain in frames:
            fh.write(struct.pack("<I", len(buf)))
            fh.write(buf)


def python_arm(frames, n_win, min_instances):
    """combdll.fleet_dll_comb, through the REAL client ring."""
    client = telem.TelemClient(host="127.0.0.1", port=0, depth=64)
    for buf, _chain in frames:
        hdr = telem._HDR.unpack_from(buf, 0)
        client._store_frame(telem.TelemFrame(hdr, buf, 0.0))
    out = {}
    for chain in client.chains():
        # lag=1 and n_win here must match how the C++ arm windowed the same stream: it closes a
        # window when a newer one arrives (so the newest is still open == lag 1) and keeps the
        # last n_win closed ones. Run the C++ side with --no-flush or the two disagree by one
        # window, which is a real difference and not a rounding one.
        out[chain] = combdll.fleet_dll_comb(client, chain, n_win=n_win, lag=1,
                                            min_instances=min_instances, k_sigma=3.0,
                                            q_fallback=2.2, per_channel=False)
    return out


def python_taps(frames, n_win):
    """combdll.instance_taps, per instance AND per channel, through the REAL client ring.

    The object the C++ `taps()` accessor now forms on the gather host. Compared field by field
    because THIS PATH HAS NO OTHER GATE: broker_equiv replays only what goes through
    gnss_broker.transport, and the telemetry stream is a raw socket, so a replay runs with no
    telemetry at all and quietly falls back to the polled discriminator. The digest stays green
    while testing none of it.
    """
    client = telem.TelemClient(host="127.0.0.1", port=0, depth=64)
    for buf, _chain in frames:
        hdr = telem._HDR.unpack_from(buf, 0)
        client._store_frame(telem.TelemFrame(hdr, buf, 0.0))
    out = {}
    for chain in client.chains():
        wins = client.windows(chain, lag=1)[-int(n_win):]
        out[chain] = combdll.instance_taps(client, chain, wins, per_channel=True)
    return out


#: Per-instance tap fields, with tolerances. The powers are means over records; `chan` is
#: compared separately because its denominator is per channel, not per instance.
TAP_FIELDS = [("e", 1e-9), ("p", 1e-9), ("l", 1e-9), ("n_chan", 1e-9),
              ("n_rec", 0.0), ("hop", 0.0)]


def compare_taps(py, cpp):
    """[(severity, message)] -- empty means the per-instance/per-channel taps agree."""
    bad = []
    cj = cpp.get("taps", {})
    for chain in sorted(set(py) | set(cj)):
        p_prns = {int(k): v for k, v in (py.get(chain) or {}).items()}
        c_prns = {int(k): v for k, v in (cj.get(chain) or {}).items()}
        # Python creates an entry for any (prn, inst) with a live comb; a PRN whose every
        # record was empty has n_rec 0 and is dropped by fleet_dll_comb, so compare on the
        # populated set and say so if the SETS differ.
        p_use = {p: {i: d for i, d in v.items() if d["n_rec"] > 0} for p, v in p_prns.items()}
        p_use = {p: v for p, v in p_use.items() if v}
        for prn in sorted(set(p_use) - set(c_prns)):
            bad.append(("TAP-MISSING", "%s PRN %d: Python has taps, C++ does not" % (chain, prn)))
        for prn in sorted(set(c_prns) - set(p_use)):
            bad.append(("TAP-EXTRA", "%s PRN %d: C++ has taps, Python does not" % (chain, prn)))
        for prn in sorted(set(p_use) & set(c_prns)):
            pi, ci = p_use[prn], c_prns[prn]
            for inst in sorted(set(pi) - set(ci)):
                bad.append(("TAP-INST", "%s PRN %d: instance %s only in Python" % (chain, prn, inst)))
            for inst in sorted(set(ci) - set(pi)):
                bad.append(("TAP-INST", "%s PRN %d: instance %s only in C++" % (chain, prn, inst)))
            for inst in sorted(set(pi) & set(ci)):
                a, b = pi[inst], ci[inst]
                for f, tol in TAP_FIELDS:
                    x, y = float(a[f]), float(b[f])
                    if abs(x - y) > tol * max(1.0, abs(x), abs(y)):
                        bad.append(("TAP", "%s PRN %d %s %s: py %.17g cpp %.17g"
                                    % (chain, prn, inst, f, x, y)))
                pc = {int(k): v for k, v in a["chan"].items()}
                cc = {int(k): v for k, v in b["chan"].items()}
                if set(pc) != set(cc):
                    bad.append(("TAP-CHAN", "%s PRN %d %s: freq_id sets differ py %s cpp %s"
                                % (chain, prn, inst, sorted(pc), sorted(cc))))
                    continue
                for fid in sorted(pc):
                    for k, name in enumerate(("e", "p", "l", "n_rec")):
                        x, y = float(pc[fid][k]), float(cc[fid][k])
                        tol = 0.0 if name == "n_rec" else 1e-9
                        if abs(x - y) > tol * max(1.0, abs(x), abs(y)):
                            bad.append(("TAP-CHAN", "%s PRN %d %s fid %d %s: py %.17g cpp %.17g"
                                        % (chain, prn, inst, fid, name, x, y)))
    return bad


def cpp_arm(exe, path, n_win, min_instances, arm=None, pol=None):
    cmd = [exe, path, "--n-win", str(n_win), "--min-instances", str(min_instances), "--no-flush"]
    if arm:
        cmd += ["--arm", ",".join(str(p) for p in sorted(arm))]
        for k in ("gain", "leak", "clamp", "spacing"):
            cmd += ["--" + k, repr((pol or {}).get(k, DEFAULT_POLICY[k]))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("fleetdll failed (%d): %s" % (r.returncode, r.stderr.strip()))
    return json.loads(r.stdout)


#: The shipped broker defaults (--dll-gain, --dll-leak-present, the +-3 clamp, --dll-spacing).
#: ⚠️ leak 0.05 puts the integrator's steady state at gain*0.25/leak = 1.25 chips, so the 3.0
#: clamp is unreachable BY CONSTRUCTION -- see combdll.dll_integrate. The gate uses the shipped
#: values deliberately: a gate run at constants nobody deploys measures a loop nobody runs.
DEFAULT_POLICY = {"gain": 0.25, "leak": 0.05, "clamp": 3.0, "spacing": 0.5}


def python_disc_series(frames, chain, wins_seen, n_win, min_instances):
    """{win: {prn: disc}} -- Python's fleet discriminator at EVERY window close, not just the last.

    F1 compared the two arms at the end of the stream. That is enough to catch a wrong fold and
    not enough to certify the INTEGRATOR, which walks the whole series: one window disagreeing in
    the middle would be invisible at the end but would have moved every trim after it. Built by
    replaying prefixes of the same bytes through the real client, which is slow and exact --
    both correct properties for a gate.
    """
    out = {}
    for w in wins_seen:
        client = telem.TelemClient(host="127.0.0.1", port=0, depth=64)
        for buf, _c in frames:
            hdr = telem._HDR.unpack_from(buf, 0)
            if hdr[9] > w:            # header field 9 is `win`
                continue
            client._store_frame(telem.TelemFrame(hdr, buf, 0.0))
        # lag=0: the C++ CLOSED w (a newer frame arrived), so w is inside its ring. Slicing to
        # the last n_win of everything <= w reproduces exactly the ring it aggregated on.
        ws = client.windows(chain, lag=0)[-n_win:]
        if not ws:
            continue
        rows = combdll.instance_taps(client, chain, ws, per_channel=False)
        got = {}
        for prn, per_inst in rows.items():
            use = {i: d for i, d in per_inst.items() if d["n_rec"] > 0}
            if len(use) < min_instances:
                continue
            # ⚠️ NO /n_rec HERE. instance_taps ALREADY divides by it (see the tail of that
            # function) -- fleet_dll_comb sums the per-instance values raw, and so must this.
            # Dividing again weights each instance by 1/n_rec^2, which moved disc by ~1e-3 and
            # read as a C++/Python disagreement. Harness bug, caught by the gate, 2026-08-15.
            E = sum(d["e"] for d in use.values())
            L = sum(d["l"] for d in use.values())
            if E + L > 0.0:
                got[prn] = (E - L) / (E + L)
        out[w] = got
    return out


# ---------------------------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------------------------

#: Fields compared, with tolerances. disc and q are what the loop ACTUATES on, so they are held
#: tightest; the powers are sums whose instance ordering differs between a Python dict and a
#: std::map, which is float noise at ~1e-16 and nothing else.
FIELDS = [("disc", 1e-9), ("q", 1e-9), ("e_pow", 1e-9), ("p_pow", 1e-9), ("l_pow", 1e-9),
          ("n_src", 0.0), ("n_rec", 0.0), ("n_chan", 1e-9), ("hop", 0.0)]


def compare(py, cpp):
    """[(severity, message)] -- empty means the two arms agree."""
    bad = []
    for chain in sorted(set(py) | set(cpp.get("chains", {}))):
        p_rows = py.get(chain, {})
        c_rows = (cpp.get("chains", {}).get(chain) or {}).get("prns", {})
        p_prns = set(int(k) for k in p_rows)
        c_prns = set(int(k) for k in c_rows)
        for prn in sorted(p_prns - c_prns):
            bad.append(("MISSING", "%s PRN %d: Python has a row, C++ does not" % (chain, prn)))
        for prn in sorted(c_prns - p_prns):
            bad.append(("EXTRA", "%s PRN %d: C++ has a row, Python does not" % (chain, prn)))
        for prn in sorted(p_prns & c_prns):
            a, b = p_rows[prn], c_rows[str(prn)]
            for key, tol in FIELDS:
                x, y = float(a[key]), float(b[key])
                d = abs(x - y)
                rel = d / max(abs(x), abs(y), 1e-30)
                if d > 0.0 and rel > tol:
                    bad.append(("DIFF", "%s PRN %d %s: py %.12g vs cpp %.12g (rel %.3g)"
                                % (chain, prn, key, x, y, rel)))
    return bad


def describe(py):
    """What the fixture actually exercised -- so a vacuous pass is visible as one."""
    lines = []
    for chain in sorted(py):
        rows = py[chain]
        if not rows:
            lines.append("  %-9s NO ROWS" % chain)
            continue
        discs = sorted(v["disc"] for v in rows.values())
        qs = sorted(v["q"] for v in rows.values())
        lines.append("  %-9s %d PRN  disc %+.3f..%+.3f  q %.2f..%.2f  n_src %d..%d"
                     % (chain, len(rows), discs[0], discs[-1], qs[0], qs[-1],
                        min(v["n_src"] for v in rows.values()),
                        max(v["n_src"] for v in rows.values())))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default=os.path.join(_HERE, "fleetdll"))
    ap.add_argument("--n-win", type=int, default=4, help="windows averaged (broker default 4)")
    ap.add_argument("--min-instances", type=int, default=2)
    ap.add_argument("--windows", type=int, default=12, help="windows in the fixture")
    ap.add_argument("--self-test", action="store_true",
                    help="ALSO perturb the fixture and require the comparison to fail")
    ap.add_argument("--keep", help="directory to leave the fixture and both answers in")
    ap.add_argument("--no-build", action="store_true",
                    help="do NOT rebuild the C++ tool first (for bisecting a binary you built "
                         "on purpose; the default builds, see the note below)")
    args = ap.parse_args()

    # ⚠️ BUILD IT. DO NOT TRUST THE BINARY THAT IS THERE (2026-08-23).
    # `fleetdll` is a SYMLINK to a per-host build (build_tool.sh writes $N.$HOSTNAME so two
    # hosts sharing this NFS tree do not clobber each other). Checking only that the path
    # exists therefore tests WHOEVER BUILT LAST, on WHATEVER SOURCE THEY HAD. Found the hard
    # way: the link pointed at a cx19 binary from 08-15, the wire format went to v5 on 08-16
    # (RECORD_FLOATS 28 -> 29), and every frame in the fixture came back BAD_HEADER. The gate
    # had been red for eight days and reported it as "Python has a row, C++ does not" -- which
    # reads as a fold disagreement, not a stale binary. This is exactly what
    # "verify what RUNS, not what you built" is about, and a GATE is the last place that
    # should get it wrong.
    if not args.no_build:
        b = subprocess.run([os.path.join(_HERE, "build_tool.sh"), "fleetdll"],
                           capture_output=True, text=True, cwd=_HERE)
        if b.returncode != 0:
            raise SystemExit("build_tool.sh fleetdll FAILED:\n%s\n%s"
                             % (b.stdout[-2000:], b.stderr[-2000:]))
    if not os.path.exists(args.exe):
        raise SystemExit("no %s -- build it with ./build_tool.sh fleetdll" % args.exe)

    tmp = args.keep or tempfile.mkdtemp(prefix="fleetdll_gate-")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "frames.bin")

    frames = build_fixture(args.windows)
    write_stream(path, frames)
    py = python_arm(frames, args.n_win, args.min_instances)
    cpp = cpp_arm(args.exe, path, args.n_win, args.min_instances)

    print("fixture: %d frames, %d windows, %d instances, %d B/frame"
          % (len(frames), args.windows, len(INSTANCES), len(frames[0][0])))
    print("         C++ read %d, closed %s, late %d"
          % (cpp["frames"],
             {k: v["windows_closed"] for k, v in cpp["chains"].items()}, cpp["late_frames"]))
    for line in describe(py):
        print(line)

    # ⚠️ AN EXPERIMENT THAT CANNOT SUCCEED IS NOT AN EXPERIMENT. Before believing a match,
    # check the fixture produced something a mismatch could show up in: several PRNs, a spread
    # of discriminators, and the min_instances exclusion actually biting.
    problems = []
    for chain, rows in py.items():
        if len(rows) < 4:
            problems.append("%s produced only %d rows" % (chain, len(rows)))
        if rows and max(v["disc"] for v in rows.values()) \
                - min(v["disc"] for v in rows.values()) < 0.2:
            problems.append("%s discriminators span < 0.2 -- the fixture is degenerate" % chain)
        if 11 in rows:
            problems.append("%s PRN 11 was NOT excluded -- min_instances is not being applied,"
                            " so the gate is not testing what it claims" % chain)
    if problems:
        for p in problems:
            print("FIXTURE PROBLEM: %s" % p)
        return 3

    bad = compare(py, cpp)
    if args.keep:
        with open(os.path.join(tmp, "python.json"), "w") as fh:
            json.dump({c: {str(k): v for k, v in r.items()} for c, r in py.items()}, fh,
                      indent=2, default=str)
        with open(os.path.join(tmp, "cpp.json"), "w") as fh:
            json.dump(cpp, fh, indent=2)

    if bad:
        print("\nFAIL -- %d disagreement(s):" % len(bad))
        for sev, msg in bad[:40]:
            print("  %-8s %s" % (sev, msg))
        if len(bad) > 40:
            print("  ... and %d more" % (len(bad) - 40))
        return 1
    n = sum(len(r) for r in py.values())
    print("\nPASS -- %d PRN rows agree across both arms on identical bytes." % n)

    # THE ONE PLACE THE ARMS ARE MEANT TO DIVERGE, asserted rather than left to be discovered.
    # A frame for an already-closed window is folded by the PYTHON store (it keys on `win` and
    # will happily amend a window still inside the ring) and DROPPED by the C++ (a control loop
    # cannot retroactively amend a step it has already taken, and re-opening a window would
    # make the aggregate depend on arrival order -- the exact inference #59 removed).
    # Frequency is a fact about the fleet, not an assumption: `late_frames` is served by
    # GnssFleetTrim/get_stats and must be watched on sky, not presumed small.
    late = frames + [frames[0]]          # replay the very first frame at the end of the stream
    p3 = os.path.join(tmp, "frames_late.bin")
    write_stream(p3, late)
    c3 = cpp_arm(args.exe, p3, args.n_win, args.min_instances)
    if c3["late_frames"] != 1:
        print("FAIL -- a frame for a long-closed window was not counted as late "
              "(late_frames %d, expected 1)." % c3["late_frames"])
        return 1
    if compare(py, c3):
        print("FAIL -- a late frame CHANGED the C++ answer. It must be dropped, not folded: "
              "folding it makes the aggregate depend on arrival order.")
        return 1
    print("LATE-FRAME PASS -- counted (1) and dropped; the answer is unchanged.")

    # ---- THE PER-INSTANCE / PER-CHANNEL TAPS -------------------------------------------
    # A tighter comparison than the discriminator legs above, and deliberately so: disc and q
    # are RATIOS, so a common factor on E, P and L cancels and a whole class of fold error
    # survives them. These are the unreduced numbers.
    pt = python_taps(frames, args.n_win)
    bad_t = compare_taps(pt, cpp)
    if bad_t:
        print("FAIL -- %d tap disagreement(s):" % len(bad_t))
        for sev, msg in bad_t[:12]:
            print("  %-11s %s" % (sev, msg))
        return 1
    n_it = sum(len(v) for ch in pt.values() for v in ch.values())
    n_ch = sum(len(d["chan"]) for ch in pt.values() for v in ch.values() for d in v.values())
    print("TAPS PASS -- %d (PRN, instance) taps and %d per-channel rows agree, e/p/l/n_chan "
          "to 1e-9 and n_rec/hop exactly." % (n_it, n_ch))

    # ---- THE INTEGRATOR (F2) -----------------------------------------------------------
    # Two legs, deliberately separate, because a single end-to-end comparison would let a fold
    # error and a recurrence error cancel:
    #   (a) the C++ disc at EVERY window close matches Python's on the same window set;
    #   (b) Python's dll_integrate, driven by that disc series, reproduces the C++ trim series.
    # (b) alone would be circular if (a) were not established; (a) alone is what F1 already did,
    # but only at the last window.
    ARM = list(SIGNAL_PRNS)   # see SIGNAL_PRNS: the noise PRNs calibrate, they do not arm
    cpp2 = cpp_arm(args.exe, path, args.n_win, args.min_instances, arm=ARM)
    series = cpp2.get("series", {}).get("gps_l5", {})
    if not series:
        print("FAIL -- the C++ integrator stepped on nothing. It was armed on %s." % ARM)
        return 1
    wins_seen = sorted({int(r[0]) for rows in series.values() for r in rows})
    pyd = python_disc_series(frames, "gps_l5", wins_seen, args.n_win, args.min_instances)

    bad_i = []
    n_disc = n_trim = 0
    for prn_s, rows in sorted(series.items()):
        prn = int(prn_s)
        trim = 0.0
        for win, disc, cpp_trim in rows:
            win = int(win)
            # (a) the discriminator, per window
            ref = pyd.get(win, {}).get(prn)
            if ref is None:
                bad_i.append("PRN %d win %d: C++ stepped, Python formed no discriminator" % (prn, win))
                continue
            if abs(ref - disc) > 1e-9 * max(1.0, abs(ref)):
                bad_i.append("PRN %d win %d disc: py %.12g vs cpp %.12g" % (prn, win, ref, disc))
            n_disc += 1
            # (b) the recurrence, driven by the SAME disc
            trim = combdll.dll_integrate(trim, disc, **DEFAULT_POLICY)
            if abs(trim - cpp_trim) > 1e-12 * max(1.0, abs(trim)):
                bad_i.append("PRN %d win %d trim: py %.12g vs cpp %.12g" % (prn, win, trim, cpp_trim))
            n_trim += 1
    if bad_i:
        print("\nFAIL -- integrator: %d disagreement(s):" % len(bad_i))
        for m in bad_i[:20]:
            print("  %s" % m)
        return 1

    # ⚠️ AND CHECK IT ACTUALLY MOVED. An integrator that returned 0 every step would pass every
    # comparison above. The trims must be non-trivial and must show the leak ceiling.
    finals = {int(p): rows[-1][2] for p, rows in series.items() if rows}
    ceiling = DEFAULT_POLICY["gain"] * 0.25 / DEFAULT_POLICY["leak"]
    if max(abs(v) for v in finals.values()) < 1e-3:
        print("FAIL -- every trim is ~0. The integrator ran but did not integrate.")
        return 1
    print("INTEGRATOR PASS -- %d window discriminators and %d integrator steps agree "
          "(%d PRNs, %d steps each)." % (n_disc, n_trim, len(series),
                                         len(next(iter(series.values())))))
    print("  final trims: %s" % ", ".join("PRN %d %+.4f" % (p, v) for p, v in sorted(finals.items())))
    print("  leak ceiling gain*0.25/leak = %.3f chips; max |trim| here %.3f"
          % (ceiling, max(abs(v) for v in finals.values())))

    if args.self_test:
        # THE GATE MUST BE ABLE TO FAIL. Nudge one channel of one record by 1% and require the
        # comparison to notice. If this passes, the comparison is not comparing.
        #
        # ⚠️ THE FRAME HAS TO BE ONE THE ANSWER DEPENDS ON. The first version poked frame 0 and
        # the self-test failed -- correctly: with n_win 4 and 12 windows the ring holds windows
        # 7..10, and window 0 fell out of it long before the end of the file. A perturbation
        # outside the averaging window is invisible BY DESIGN, and reading that as "the fold is
        # broken" would have been a wasted afternoon. Window W-2 is the NEWEST CLOSED window,
        # so it is in the ring for any n_win >= 1.
        n_per_win = 2 * len(INSTANCES)  # two chains
        poke_at = (args.windows - 2) * n_per_win
        buf = bytearray(frames[poke_at][0])
        # ⚠️ POKE A PRN THAT CONTRIBUTES -- DO NOT POKE A FIXED ROW (2026-08-23).
        # This used to poke row 0, whose PRN is whatever the fixture's SHUFFLE put there
        # (row compaction, #64). It landed on PRN 11 -- the row that exists precisely to be
        # EXCLUDED by min_instances -- so the "perturbation" was applied to the one satellite
        # guaranteed not to reach the answer, and the self-test reported the comparison as
        # insensitive when it was the poke that was inert. A self-test that can be defeated by
        # reordering the fixture is not a self-test. Find the row by PRN.
        _row = None
        for _r in range(N_PRN):
            _o = telem._HDR_BYTES + _r * telem._ROW_TOTAL * 4
            if int(struct.unpack_from("<f", buf, _o)[0] + 0.5) in SIGNAL_PRNS:
                _row = _r
                break
        if _row is None:
            raise SystemExit("self-test: no SIGNAL_PRNS row in the poked frame -- the fixture "
                             "changed shape and the poke would be inert")
        off = (telem._HDR_BYTES + _row * telem._ROW_TOTAL * 4
               + telem._ROW_FLOATS * 4 + telem.CHAN_E_RE * 4)
        (v,) = struct.unpack_from("<f", buf, off)
        struct.pack_into("<f", buf, off, v * 1.01 + 1.0)
        poked = frames[:poke_at] + [(bytes(buf), frames[poke_at][1])] + frames[poke_at + 1:]
        p2 = os.path.join(tmp, "frames_poked.bin")
        write_stream(p2, poked)
        bad2 = compare(python_arm(poked, args.n_win, args.min_instances),
                       cpp_arm(args.exe, p2, args.n_win, args.min_instances))
        # Both arms see the SAME poked bytes, so they must still AGREE with each other -- the
        # self-test proves the arms track each other, and the check below proves the comparison
        # is sensitive at all by requiring the poked answer to differ from the clean one.
        if bad2:
            print("SELF-TEST FAIL: the arms disagree on the perturbed fixture: %s" % bad2[:3])
            return 1
        moved = compare(py, cpp_arm(args.exe, p2, args.n_win, args.min_instances))
        if not moved:
            print("SELF-TEST FAIL: a 1%% perturbation of one channel (window %d, the newest "
                  "closed one) changed NOTHING -- the comparison, the fixture, or the fold is "
                  "not doing what it claims." % (args.windows - 2))
            return 1
        print("SELF-TEST PASS -- the arms still agree on perturbed bytes, and the comparison "
              "sees the perturbation (%d field(s) moved)." % len(moved))
    if not args.keep:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
    else:
        print("kept in %s" % tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
