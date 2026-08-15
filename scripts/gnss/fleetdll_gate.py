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
           16: (0.00, 0.02)}   # essentially noise


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
                prns = [4, 9, 27, 3, 16]
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


def cpp_arm(exe, path, n_win, min_instances):
    r = subprocess.run([exe, path, "--n-win", str(n_win), "--min-instances",
                        str(min_instances), "--no-flush"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("fleetdll failed (%d): %s" % (r.returncode, r.stderr.strip()))
    return json.loads(r.stdout)


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
    args = ap.parse_args()

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
        off = telem._HDR_BYTES + telem._ROW_FLOATS * 4 + telem.CHAN_E_RE * 4
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
            print("SELF-TEST FAIL: a 1% perturbation of one channel (window %d, the newest "
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
