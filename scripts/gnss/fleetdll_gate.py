#!/usr/bin/env python3
"""Gate: does the C++ fleet discriminator agree with the Python one?

Builds a fixture of telemetry frames, hands the same bytes to both arms, and compares the
discriminator they produce for every PRN:

    Python:  gnss_broker.telem.TelemFrame -> TelemClient._store_frame -> combdll.fleet_dll_comb
    C++:     scripts/gnss/fleetdll  ->  gnss::FleetDll  (lib/stages/gnss/gnssFleetDll.hpp)

Neither arm is a model of the other: both are the shipped code. The Python side goes through
the real client ring (so `windows(lag=1)` and the eviction policy are exercised, not
reimplemented), and the C++ side is the same header GnssFleetTrim runs in production. Further
legs compare the per-PRN and per-channel taps, the per-record series and the integrator's trim
series, and check xcoh, the C++ arm's late-frame drop, taps_win != n_win, the trim store and
the absolute floor.

Bytes, not numbers: a defect in the convention between two components (what a number means,
what it is reduced modulo, which epoch it references) is invisible to a test of either alone.
A fixture of floats handed to two parsers tests the parsers; a fixture of bytes tests the
contract.

    ./fleetdll_gate.py                 # build, run both arms, compare
    ./fleetdll_gate.py --self-test     # also prove the comparison can fail (see below)
    ./fleetdll_gate.py --keep DIR      # leave the fixture and both answers behind

The gate must be able to fail. --self-test perturbs one channel's early tap in one record
(x1.01 + 1.0) and requires both arms to still agree with each other while differing from the
clean answer, then moves one sender's REC_PHI0 and requires the lobe power to move.

The coherence unit is the lobe, never the sender. Each sender's rows carry REC_PHI0, the
per-sender phase reference the assembler applied, and the fixture writes a different one per
sender, record and PRN, rotating that sender's comb by exp(-i*phi0) so that only the rotation
by exp(+i*phi0) recovers the common reference. An arm that dropped the rotation, or formed a
power per sender before adding, gives a different lobe power here. Absolute freq_ids are
disjoint per sender (stride 8, offset by the sender index) except one deliberate duplicate, so
the per-channel table's duplicate handling is exercised too.

The fixture is large enough for the shapes a bug hides in (row compaction, instances with
different channel counts, a PRN below min_instances), which do not appear at three PRNs.
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
# The fixture. Built from the same struct the broker parses with (telem._HDR), so a layout
# drift fails here as a struct error rather than as quietly wrong numbers.
# ---------------------------------------------------------------------------------------------

N_REC = 4  # records per frame, as in production
N_PRN = 16  # rows per record (production chains carry ~31)
HOPS_PER_RECORD = 2048
FFT_LEN = 16384
SPACING = 0.5  # E/L spacing, chips

#: (instance, n_chan). Instances carry different channel counts on purpose: n_chan is a
#: per-sender header field, and a consumer that assumes one value reads the next row's floats.
INSTANCES = [
    ("cx19.0", 7),
    ("cx19.1", 6),
    ("cx27.0", 7),
    ("cx27.1", 8),
    ("cx42.0", 7),
    ("cx42.1", 6),
]

#: The freq_id one sender carries that another also does. A freq_id is absolute and belongs to
#: exactly one sender per window; two senders both carrying it is a routing fault upstream that
#: the per-channel table must drop (and count) rather than average, while the lobe sum keeps it.
DUP_FID = 5960


def chan_ids_for(i, n_chan):
    """A sender's absolute freq_ids: stride 8, offset by the sender index, so disjoint per
    sender, plus DUP_FID on the last sender (already the first sender's first channel)."""
    ids = [5960 + i + 8 * k for k in range(n_chan)]
    if i == len(INSTANCES) - 1:
        ids[-1] = DUP_FID
    return ids


def phi0_for(i, r, prn):
    """The per-sender phase reference: arbitrary per sender, record and PRN, as on the fleet."""
    return 0.9 * i + 0.37 * r + 0.11 * prn


#: prn -> (code offset in chips, amplitude). Offset 0 = on peak.
#: Spread across the pull-in region so `disc` is not a single value: a gate whose fixture
#: produces one number cannot tell a working discriminator from a constant.
TARGETS = {
    4: (0.00, 1.00),  # on peak: disc ~ 0, q high
    9: (0.22, 0.80),  # early shoulder
    27: (-0.31, 0.65),  # late shoulder
    3: (0.55, 0.40),  # far out
    11: (0.10, 0.50),  # one instance only -> must be excluded by min_instances
    20: (0.15, 0.60),  # two instances on odd windows, one on even: complete in half
    # the windows, so the record-level gate must bite per window
    16: (0.00, 0.02),  # essentially noise
    # Near-noise rows (amplitude ~0.02), as on sky, where most of a chain's rows carry little
    # or no signal at any moment. They are never armed (SIGNAL_PRNS).
    5: (0.00, 0.02),
    12: (0.00, 0.015),
    18: (0.00, 0.025),
    21: (0.00, 0.018),
    24: (0.00, 0.022),
    30: (0.00, 0.017),
}

#: PRNs the integrator legs arm: real signal only. 11 is the min_instances exclusion; the
#: near-noise PRNs are not armed, since arming them would ask the loop to step on noise.
SIGNAL_PRNS = [3, 4, 9, 27]


def _R(x):
    """Triangular code correlation, chips. The fixture's only physics."""
    return max(0.0, 1.0 - abs(x))


def _chan_block(rng, prn, fid, dead_energy, phi0, ramp=0.7, amp_scale=1.0):
    """One channel's nine floats, prompt then early then late (re, im, replica energy each),
    raw as the assembler writes them, on the sender's reference: the common phase rotated by
    exp(-i*phi0)."""
    d, amp = TARGETS[prn]
    amp *= amp_scale
    if dead_energy:
        # No live comb on this channel this record: the assembler leaves the energy at zero
        # and the consumer must skip it; averaging the zeros in would dilute the power.
        return [0.0] * 9
    e_p = 40.0 + 3.0 * (fid % 5)  # prompt replica energy, varying per channel
    phi = ramp * fid + 0.3 * prn - phi0  # a per-channel phase, so |sum| != sum|.|
    noise = lambda: rng.gauss(0.0, 0.02 * amp + 0.01)

    def tap(off, energy):
        a = amp * _R(d + off) + noise()
        return [a * energy * math.cos(phi), a * energy * math.sin(phi), energy]

    e_e = e_p * 0.98
    e_l = e_p * 1.02
    p = tap(0.0, e_p)
    e = tap(+SPACING, e_e)
    l = tap(-SPACING, e_l)
    return [p[0], p[1], p[2], e[0], e[1], e_e, l[0], l[1], e_l]


def build_frame(
    chain,
    inst,
    inst_idx,
    n_chan,
    win,
    seq,
    prn_rows,
    present,
    rng,
    dead,
    ramp=0.7,
    amp_scale=1.0,
):
    """One wire frame, bytes. `prn_rows` is the row->PRN map (row compaction)."""
    # The sender's own stride, as a real sender builds it: the row carries this instance's
    # comb columns and no reserved ones. The senders here have different n_chan, so a fold
    # that strides by the format's ceiling reads most of them at the wrong offsets.
    row_total = telem._ROW_FLOATS + n_chan * telem._CHAN_FLOATS
    wstart0 = win * N_REC * HOPS_PER_RECORD * FFT_LEN
    chan_ids = chan_ids_for(inst_idx, n_chan) + [0] * (telem._MAX_CHAN - n_chan)
    hdr = telem._HDR.pack(
        telem._MAGIC,
        telem._VERSION,
        N_REC,
        N_PRN,
        telem._ROW_FLOATS,
        n_chan,
        32,
        HOPS_PER_RECORD,
        FFT_LEN,
        win,
        seq,
        wstart0,
        0.0,
        present,
        n_chan,
        row_total,
        chain.encode(),
        inst.encode(),
        *chan_ids
    )

    rows = [0.0] * (N_REC * N_PRN * row_total)
    for r in range(N_REC):
        for p, prn in enumerate(prn_rows):
            base = (r * N_PRN + p) * row_total
            # PRN is written on every row of every record, including records that did not
            # run, so the row map is read from the data instead of configured.
            rows[base + telem.REC_PRN] = float(prn)
            if prn == 0 or not (present & (1 << r)):
                continue
            phi0 = phi0_for(inst_idx, r, prn)
            rows[base + telem.REC_PHI0] = phi0
            # Projection cost (REC_PROJ_COST), a per-sender pattern so the (1 - b) weight path
            # is exercised in both arms: no basis (-1) on instance 0, graded on the others.
            rows[base + telem.REC_PROJ_COST] = (
                -1.0
                if inst_idx == 0
                else 0.3
                if prn % 2
                else 0.05 * ((r + inst_idx) % 3)
            )
            for ch in range(n_chan):
                cb = base + telem._ROW_FLOATS + ch * telem._CHAN_FLOATS
                blk = _chan_block(
                    rng,
                    prn,
                    chan_ids[ch],
                    dead_energy=(ch, prn) in dead["energy"],
                    phi0=phi0,
                    ramp=ramp,
                    amp_scale=amp_scale,
                )
                rows[cb : cb + telem._CHAN_FLOATS] = blk
    return hdr + struct.pack("<%df" % len(rows), *rows)


def build_fixture(n_win, seed=20260815, ramp=0.7, amp_scale=1.0):
    """[(bytes, chain)] in delivery order: window-major, as the live path sees them.

    `ramp` is the per-channel phase slope (rad per freq_id); `amp_scale` multiplies every
    PRN's amplitude. Both exist for the XCOH leg, which needs a stream whose senders are
    exactly co-phased (ramp 0) and one that is pure noise (amp_scale 0); the main fixture
    keeps both at their defaults."""
    rng = random.Random(seed)
    dead = {"energy": {(2, 9), (5, 27)}}  # two channels with no live comb
    frames = []
    for win in range(n_win):
        for chain in ("gps_l5", "gal_e5a"):
            for i, (inst, n_chan) in enumerate(INSTANCES):
                # Row compaction: the row->PRN map differs per instance and is not the
                # configured PRN order. An arm that assumed row == PRN index would read another
                # satellite's comb and produce a confident wrong discriminator.
                prns = [4, 9, 27, 3, 16, 5, 12, 18, 21, 24, 30]
                if i == 0:
                    prns = prns + [11, 20]  # PRN 11 on one instance only
                if i == 1 and win % 2:
                    prns = prns + [20]  # PRN 20 reaches two only on odd windows
                order = list(prns)
                rng.shuffle(order)
                prn_rows = order + [0] * (N_PRN - len(order))
                # A record slot that did not run is a hole at a known index. Vary which.
                present = 0xF if (win + i) % 4 else 0xD
                frames.append(
                    (
                        build_frame(
                            chain,
                            inst,
                            i,
                            n_chan,
                            win,
                            win * 100 + i,
                            prn_rows,
                            present,
                            rng,
                            dead,
                            ramp=ramp,
                            amp_scale=amp_scale,
                        ),
                        chain,
                    )
                )
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
    """combdll.fleet_dll_comb, through the real client ring."""
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
        out[chain] = combdll.fleet_dll_comb(
            client,
            chain,
            n_win=n_win,
            lag=1,
            min_instances=min_instances,
            k_sigma=3.0,
            q_fallback=2.2,
        )
    return out


def python_taps(frames, n_win, min_instances):
    """combdll.lobe_taps, per PRN and per channel, through the real client ring.

    The object the C++ `taps()` accessor forms on the gather host, compared field by field
    because no other gate covers this path: broker_equiv replays only what goes through
    gnss_broker.transport, and telemetry arrives on a raw socket, so its replays carry none.
    """
    client = telem.TelemClient(host="127.0.0.1", port=0, depth=64)
    for buf, _chain in frames:
        hdr = telem._HDR.unpack_from(buf, 0)
        client._store_frame(telem.TelemFrame(hdr, buf, 0.0))
    out = {}
    for chain in client.chains():
        wins = client.windows(chain, lag=1)[-int(n_win) :]
        out[chain] = combdll.lobe_taps(
            client, chain, wins, per_channel=True, min_instances=min_instances
        )
    return out


#: Per-PRN lobe tap fields, with tolerances. The powers are means over records; `chan` is
#: compared separately because its denominator is one channel's energy, not the lobe's.
TAP_FIELDS = [
    ("e", 1e-9),
    ("p", 1e-9),
    ("l", 1e-9),
    ("n_chan", 1e-9),
    ("n_rec", 0.0),
    ("n_inst", 0.0),
    ("hop", 0.0),
    ("xcoh", 1e-9),
    ("n_xcoh", 0.0),
]


def compare_taps(py, cpp):
    """[(severity, message)] -- empty means the per-PRN/per-channel taps agree."""
    bad = []
    cj = cpp.get("taps", {})
    for chain in sorted(set(py) | set(cj)):
        p_prns = {int(k): v for k, v in (py.get(chain) or {}).items() if v["n_rec"] > 0}
        c_prns = {int(k): v for k, v in (cj.get(chain) or {}).items()}
        for prn in sorted(set(p_prns) - set(c_prns)):
            bad.append(
                (
                    "TAP-MISSING",
                    "%s PRN %d: Python has taps, C++ does not" % (chain, prn),
                )
            )
        for prn in sorted(set(c_prns) - set(p_prns)):
            bad.append(
                ("TAP-EXTRA", "%s PRN %d: C++ has taps, Python does not" % (chain, prn))
            )
        for prn in sorted(set(p_prns) & set(c_prns)):
            a, b = p_prns[prn], c_prns[prn]
            for f, tol in TAP_FIELDS:
                if a.get(f) is None or b.get(f) is None:
                    # xcoh is None/null below two senders -- both arms must say so together
                    if a.get(f) is not None or b.get(f) is not None:
                        bad.append(
                            (
                                "TAP",
                                "%s PRN %d %s: py %r cpp %r"
                                % (chain, prn, f, a.get(f), b.get(f)),
                            )
                        )
                    continue
                x, y = float(a[f]), float(b[f])
                if abs(x - y) > tol * max(1.0, abs(x), abs(y)):
                    bad.append(
                        (
                            "TAP",
                            "%s PRN %d %s: py %.17g cpp %.17g" % (chain, prn, f, x, y),
                        )
                    )
            pc = {int(k): v for k, v in a["chan"].items()}
            cc = {int(k): v for k, v in b["chan"].items()}
            if set(pc) != set(cc):
                bad.append(
                    (
                        "TAP-CHAN",
                        "%s PRN %d: freq_id sets differ py %s cpp %s"
                        % (chain, prn, sorted(pc), sorted(cc)),
                    )
                )
                continue
            if DUP_FID in pc:
                bad.append(
                    (
                        "TAP-DUP",
                        "%s PRN %d: duplicated freq_id %d was AVERAGED into the "
                        "per-channel table" % (chain, prn, DUP_FID),
                    )
                )
            for fid in sorted(pc):
                for k, name in enumerate(("e", "p", "l", "n_rec")):
                    x, y = float(pc[fid][k]), float(cc[fid][k])
                    tol = 0.0 if name == "n_rec" else 1e-9
                    if abs(x - y) > tol * max(1.0, abs(x), abs(y)):
                        bad.append(
                            (
                                "TAP-CHAN",
                                "%s PRN %d fid %d %s: py %.17g cpp %.17g"
                                % (chain, prn, fid, name, x, y),
                            )
                        )
    return bad


def python_recs(frames, n_win):
    """combdll.lobe_records, prompt_cn0's `recs`: {(win, slot): {prn: [e, p, l, n_inst,
    n_chan]}}, through the real client ring.

    The estimator's own input, so the gate compares what it consumes and not a paraphrase.
    The statistics prompt_cn0 applies on top stay in Python and are outside this leg.
    """
    client = telem.TelemClient(host="127.0.0.1", port=0, depth=64)
    for buf, _chain in frames:
        hdr = telem._HDR.unpack_from(buf, 0)
        client._store_frame(telem.TelemFrame(hdr, buf, 0.0))
    out = {}
    for chain in client.chains():
        wins = client.windows(chain, lag=1)[-int(n_win) :]
        out[chain] = combdll.lobe_records(client, chain, wins)
    return out


def compare_recs(py, cpp):
    """[(severity, message)] -- empty means the per-record series agree."""
    bad = []
    cj = cpp.get("rec_series", {})
    for chain in sorted(set(py) | set(cj)):
        p_rows = py.get(chain) or {}
        c_rows = {}
        for win, slot, prn, n_inst, n_chan, e, p, l in cj.get(chain) or []:
            c_rows[(int(win), int(slot), int(prn))] = [
                e,
                p,
                l,
                int(n_inst),
                int(n_chan),
            ]
        p_flat = {
            (int(w), int(r), int(prn)): v
            for (w, r), per in p_rows.items()
            for prn, v in per.items()
        }
        for k in sorted(set(p_flat) - set(c_rows))[:5]:
            bad.append(
                (
                    "REC-MISSING",
                    "%s (win %d, slot %d, PRN %d) only in Python" % ((chain,) + k),
                )
            )
        for k in sorted(set(c_rows) - set(p_flat))[:5]:
            bad.append(
                (
                    "REC-EXTRA",
                    "%s (win %d, slot %d, PRN %d) only in C++" % ((chain,) + k),
                )
            )
        for k in sorted(set(p_flat) & set(c_rows)):
            a, b = p_flat[k], c_rows[k]
            for i, name in enumerate(("e", "p", "l", "n_inst", "n_chan")):
                x, y = float(a[i]), float(b[i])
                tol = 0.0 if name in ("n_inst", "n_chan") else 1e-9
                if abs(x - y) > tol * max(1.0, abs(x), abs(y)):
                    bad.append(
                        (
                            "REC",
                            "%s (win %d, slot %d, PRN %d) %s: py %.17g cpp %.17g"
                            % (chain, k[0], k[1], k[2], name, x, y),
                        )
                    )
    return bad


def cpp_arm(
    exe, path, n_win, min_instances, arm=None, pol=None, taps_win=None, trim_in=None
):
    cmd = [
        exe,
        path,
        "--n-win",
        str(n_win),
        "--min-instances",
        str(min_instances),
        "--no-flush",
    ]
    if taps_win:
        cmd += ["--taps-win", str(taps_win)]
    if trim_in:
        cmd += ["--trim-in", trim_in]
    if arm:
        cmd += ["--arm", ",".join(str(p) for p in sorted(arm))]
        for k in ("gain", "leak", "clamp", "spacing"):
            cmd += ["--" + k, repr((pol or {}).get(k, DEFAULT_POLICY[k]))]
        # Always passed: an omitted floor (0) means refuse, every armed PRN leak-only with
        # n_steps 0. The baseline is GATE_NOBODY, a floor below every power in the fixture.
        cmd += ["--p-floor-abs", repr((pol or {}).get("p_floor_abs", GATE_NOBODY))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("fleetdll failed (%d): %s" % (r.returncode, r.stderr.strip()))
    return json.loads(r.stdout)


#: A floor below every power in the fixture: it gates nobody, so it is the neutral baseline the
#: other legs are compared against. Not zero: zero means refuse everything.
GATE_NOBODY = 1e-30

#: The loop recurrence's constants, what dll_integrate takes, kept separate from the gate. They
#: are TrimPolicy's defaults and the broker's slow-DLL per-update values (--dll-gain,
#: cli._FROZEN's dll_leak_present, the +-3 clamp, --dll-spacing); the live fast loop instead
#: converts the broker's per-second values at its measured close rate. Leak 0.05 caps the
#: steady state at gain*0.25/leak = 1.25 chips, so integration alone never reaches the 3.0
#: clamp (see combdll.dll_integrate).
LOOP_POLICY = {"gain": 0.25, "leak": 0.05, "clamp": 3.0, "spacing": 0.5}

DEFAULT_POLICY = dict(LOOP_POLICY, p_floor_abs=GATE_NOBODY)


def python_disc_series(frames, chain, wins_seen, n_win, min_instances):
    """{win: {prn: disc}}: Python's fleet discriminator at every window close, not just the last.

    Comparing the arms at the end of the stream catches a wrong fold but does not certify the
    integrator, which walks the whole series: one window disagreeing mid-stream is invisible at
    the end yet moves every trim after it. Built by replaying prefixes of the same bytes through
    the real client: slow and exact.
    """
    out = {}
    for w in wins_seen:
        client = telem.TelemClient(host="127.0.0.1", port=0, depth=64)
        for buf, _c in frames:
            hdr = telem._HDR.unpack_from(buf, 0)
            if hdr[9] > w:  # header field 9 is `win`
                continue
            client._store_frame(telem.TelemFrame(hdr, buf, 0.0))
        # lag=0: the C++ closed w (a newer frame arrived), so w is inside its ring. Slicing to
        # the last n_win of everything <= w reproduces the ring it aggregated on.
        ws = client.windows(chain, lag=0)[-n_win:]
        if not ws:
            continue
        rows = combdll.lobe_taps(
            client, chain, ws, per_channel=False, min_instances=min_instances
        )
        got = {}
        for prn, t in rows.items():
            if t["n_rec"] > 0 and t["e"] + t["l"] > 0.0:
                got[prn] = (t["e"] - t["l"]) / (t["e"] + t["l"])
        out[w] = got
    return out


# ---------------------------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------------------------

#: Fields compared, with relative tolerances: 1e-9 for the floats (n_chan is a mean), whose
#: summation order can differ between a Python dict and a std::map at the ~1e-16 level, and
#: exact for n_src, n_rec and hop.
FIELDS = [
    ("disc", 1e-9),
    ("q", 1e-9),
    ("e_pow", 1e-9),
    ("p_pow", 1e-9),
    ("l_pow", 1e-9),
    ("n_src", 0.0),
    ("n_rec", 0.0),
    ("n_chan", 1e-9),
    ("hop", 0.0),
]


def compare(py, cpp):
    """[(severity, message)] -- empty means the two arms agree."""
    bad = []
    for chain in sorted(set(py) | set(cpp.get("chains", {}))):
        p_rows = py.get(chain, {})
        c_rows = (cpp.get("chains", {}).get(chain) or {}).get("prns", {})
        p_prns = set(int(k) for k in p_rows)
        c_prns = set(int(k) for k in c_rows)
        for prn in sorted(p_prns - c_prns):
            bad.append(
                ("MISSING", "%s PRN %d: Python has a row, C++ does not" % (chain, prn))
            )
        for prn in sorted(c_prns - p_prns):
            bad.append(
                ("EXTRA", "%s PRN %d: C++ has a row, Python does not" % (chain, prn))
            )
        for prn in sorted(p_prns & c_prns):
            a, b = p_rows[prn], c_rows[str(prn)]
            for key, tol in FIELDS:
                x, y = float(a[key]), float(b[key])
                d = abs(x - y)
                rel = d / max(abs(x), abs(y), 1e-30)
                if d > 0.0 and rel > tol:
                    bad.append(
                        (
                            "DIFF",
                            "%s PRN %d %s: py %.12g vs cpp %.12g (rel %.3g)"
                            % (chain, prn, key, x, y, rel),
                        )
                    )
    return bad


def describe(py):
    """What the fixture actually exercised, so a vacuous pass is visible as one."""
    lines = []
    for chain in sorted(py):
        rows = py[chain]
        if not rows:
            lines.append("  %-9s NO ROWS" % chain)
            continue
        discs = sorted(v["disc"] for v in rows.values())
        qs = sorted(v["q"] for v in rows.values())
        lines.append(
            "  %-9s %d PRN  disc %+.3f..%+.3f  q %.2f..%.2f  n_src %d..%d"
            % (
                chain,
                len(rows),
                discs[0],
                discs[-1],
                qs[0],
                qs[-1],
                min(v["n_src"] for v in rows.values()),
                max(v["n_src"] for v in rows.values()),
            )
        )
    return lines


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--exe", default=os.path.join(_HERE, "fleetdll"))
    ap.add_argument(
        "--n-win", type=int, default=4, help="windows averaged (broker default 4)"
    )
    ap.add_argument("--min-instances", type=int, default=2)
    ap.add_argument("--windows", type=int, default=12, help="windows in the fixture")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="ALSO perturb the fixture and require the comparison to fail",
    )
    ap.add_argument("--keep", help="directory to leave the fixture and both answers in")
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="do NOT rebuild the C++ tool first (for bisecting a binary you built "
        "on purpose; the default builds, see the note below)",
    )
    args = ap.parse_args()

    # Build first rather than trust the binary that is there: `fleetdll` is a symlink to a
    # per-host build (build_tool.sh writes $N.$HOST so hosts sharing this NFS tree do not
    # clobber each other), so an existing path only says someone built it, from some source.
    # A binary on an older wire format rejects every frame as BAD_HEADER, which this gate
    # reports as "Python has a row, C++ does not", not as a stale binary.
    if not args.no_build:
        b = subprocess.run(
            [os.path.join(_HERE, "site", "build_tool.sh"), "fleetdll"],
            capture_output=True,
            text=True,
            cwd=_HERE,
        )
        if b.returncode != 0:
            raise SystemExit(
                "site/build_tool.sh fleetdll FAILED:\n%s\n%s"
                % (b.stdout[-2000:], b.stderr[-2000:])
            )
    if not os.path.exists(args.exe):
        raise SystemExit(
            "no %s -- build it with site/build_tool.sh fleetdll" % args.exe
        )

    tmp = args.keep or tempfile.mkdtemp(prefix="fleetdll_gate-")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "frames.bin")

    frames = build_fixture(args.windows)
    write_stream(path, frames)
    py = python_arm(frames, args.n_win, args.min_instances)
    cpp = cpp_arm(args.exe, path, args.n_win, args.min_instances)

    print(
        "fixture: %d frames, %d windows, %d instances, %d B/frame"
        % (len(frames), args.windows, len(INSTANCES), len(frames[0][0]))
    )
    print(
        "         C++ read %d, closed %s, late %d"
        % (
            cpp["frames"],
            {k: v["windows_closed"] for k, v in cpp["chains"].items()},
            cpp["late_frames"],
        )
    )
    for line in describe(py):
        print(line)

    # Before believing a match, check that the fixture produced something a mismatch could
    # show in: several PRNs, a spread of discriminators, and the min_instances exclusion biting.
    problems = []
    for chain, rows in py.items():
        if len(rows) < 4:
            problems.append("%s produced only %d rows" % (chain, len(rows)))
        if (
            rows
            and max(v["disc"] for v in rows.values())
            - min(v["disc"] for v in rows.values())
            < 0.2
        ):
            problems.append(
                "%s discriminators span < 0.2 -- the fixture is degenerate" % chain
            )
        if 11 in rows:
            problems.append(
                "%s PRN 11 was NOT excluded -- min_instances is not being applied,"
                " so the gate is not testing what it claims" % chain
            )
        # PRN 20 is complete on odd windows only: it must be present, and with fewer records
        # than an always-complete PRN. Present with as many says the gate is per PRN, not per
        # record; absent says it is per PRN the other way round.
        if 20 not in rows or 4 not in rows:
            problems.append(
                "%s PRN 20 or PRN 4 missing -- the half-complete PRN did not reach"
                " the answer" % chain
            )
        elif not 0 < rows[20]["n_rec"] < rows[4]["n_rec"]:
            problems.append(
                "%s PRN 20 n_rec %d vs PRN 4 %d -- the record-level completeness"
                " gate did not bite PER RECORD"
                % (chain, rows[20]["n_rec"], rows[4]["n_rec"])
            )
    if problems:
        for p in problems:
            print("FIXTURE PROBLEM: %s" % p)
        return 3

    bad = compare(py, cpp)
    if args.keep:
        with open(os.path.join(tmp, "python.json"), "w") as fh:
            json.dump(
                {c: {str(k): v for k, v in r.items()} for c, r in py.items()},
                fh,
                indent=2,
                default=str,
            )
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

    # The one place the arms are meant to diverge, asserted here. A frame for an already-closed
    # window is folded by the Python store (it keys on `win` and amends a window still inside
    # the ring) and dropped by the C++ (a control loop cannot amend a step it has already
    # taken, and reopening a window would make the aggregate depend on arrival order). How
    # often it happens is a fact about the fleet: GnssFleetTrim's /get_stats serves
    # `late_frames`.
    late = frames + [frames[0]]  # replay the very first frame at the end of the stream
    p3 = os.path.join(tmp, "frames_late.bin")
    write_stream(p3, late)
    c3 = cpp_arm(args.exe, p3, args.n_win, args.min_instances)
    if c3["late_frames"] != 1:
        print(
            "FAIL -- a frame for a long-closed window was not counted as late "
            "(late_frames %d, expected 1)." % c3["late_frames"]
        )
        return 1
    if compare(py, c3):
        print(
            "FAIL -- a late frame CHANGED the C++ answer. It must be dropped, not folded: "
            "folding it makes the aggregate depend on arrival order."
        )
        return 1
    print("LATE-FRAME PASS -- counted (1) and dropped; the answer is unchanged.")

    # ---- The per-PRN / per-channel lobe taps ----------------------------------------------
    # A tighter comparison than the discriminator legs above: disc and q are ratios, so a
    # common factor on E, P and L cancels and a whole class of fold error survives them. These
    # are the unreduced numbers.
    pt = python_taps(frames, args.n_win, args.min_instances)
    bad_t = compare_taps(pt, cpp)
    if bad_t:
        print("FAIL -- %d tap disagreement(s):" % len(bad_t))
        for sev, msg in bad_t[:12]:
            print("  %-11s %s" % (sev, msg))
        return 1
    n_it = sum(len(v) for v in pt.values())
    n_ch = sum(len(d["chan"]) for v in pt.values() for d in v.values())
    n_dup = {c: v.get("dup_chan") for c, v in cpp["chains"].items()}
    closed = {c: v["windows_closed"] for c, v in cpp["chains"].items()}
    if n_dup != closed:
        print(
            "FAIL -- one freq_id is carried by two senders in EVERY window, so dup_chan must "
            "equal windows_closed: dup %s vs closed %s" % (n_dup, closed)
        )
        return 1
    if any(d["chan_dup"] != [DUP_FID] for v in pt.values() for d in v.values()):
        print(
            "FAIL -- Python did not list freq_id %d as duplicated on every PRN: %s"
            % (
                DUP_FID,
                {c: {p: d["chan_dup"] for p, d in v.items()} for c, v in pt.items()},
            )
        )
        return 1
    print(
        "TAPS PASS -- %d PRN lobe taps and %d per-channel rows agree, e/p/l/n_chan to 1e-9 "
        "and n_rec/n_inst/hop exactly; freq_id %d (two senders) dropped from the channel "
        "table and counted (%s)." % (n_it, n_ch, DUP_FID, n_dup)
    )

    # ---- xcoh means what it claims -----------------------------------------------------------
    # TAPS PASS says the arms agree on xcoh, not that xcoh is the statistic it claims to be.
    # Three streams settle that, each through both arms:
    #   co-phased  -- ramp 0: every sender's prompt sits on the common reference after the
    #                 exp(+i*phi0) rotation, so every PRN with signal must read ~1;
    #   noise      -- amp_scale 0: unrelated phasors, so every PRN must read ~0 (the main
    #                 fixture's "noise" PRNs are 2-sigma signals, not noise);
    #   reference loss -- the co-phased stream with one sender's phi0 moved pi on every row
    #                 of one PRN: that PRN's xcoh must fall, and only that PRN's.
    # The main fixture itself is not used: its within-sender ramp gives senders of different
    # channel counts different partial-sum phases, which is real arithmetic but not a
    # reference the statistic is meant to score.
    def _xc(frames):
        pth = os.path.join(tmp, "frames_xcoh.bin")
        write_stream(pth, frames)
        cpp_t = cpp_arm(args.exe, pth, args.n_win, args.min_instances)
        py_t = python_taps(frames, args.n_win, args.min_instances)
        bad = compare_taps(py_t, cpp_t)
        if bad:
            print("FAIL -- arms disagree on an XCOH stream: %s" % bad[:3])
            return None
        return {(c, p): d.get("xcoh") for c, v in py_t.items() for p, d in v.items()}

    co = _xc(build_fixture(args.windows, ramp=0.0))
    nz = _xc(build_fixture(args.windows, amp_scale=0.0))
    if co is None or nz is None:
        return 1
    if any(v is None for v in co.values()):
        print(
            "FAIL -- xcoh is None on a PRN two or more senders reached: %s"
            % sorted(k for k, v in co.items() if v is None)
        )
        return 1
    lo_sig = min(v for (c, p), v in co.items() if TARGETS[p][1] >= 0.4)
    hi_nz = max(abs(v) for v in nz.values() if v is not None)
    if lo_sig < 0.9 or hi_nz > 0.3:
        print(
            "FAIL -- xcoh does not read the reference: min over co-phased signal PRNs %.3f "
            "(want >= 0.9), max |xcoh| over pure-noise PRNs %.3f (want <= 0.3)"
            % (lo_sig, hi_nz)
        )
        return 1
    # the reference loss: sender 0's phi0 moved pi on every row of PRN 4, co-phased stream
    lost = []
    for buf, chain in build_fixture(args.windows, ramp=0.0):
        b = bytearray(buf)
        inst = telem._HDR.unpack_from(b)[17]
        if inst.rstrip(b"\0") == INSTANCES[0][0].encode():
            hdr = telem._HDR.unpack_from(b)
            n_rec, n_prn, row_total = hdr[2], hdr[3], hdr[15]
            for r in range(n_rec):
                for pi in range(n_prn):
                    base = telem._HDR_BYTES + (r * n_prn + pi) * row_total * 4
                    (prn_f,) = struct.unpack_from("<f", b, base + telem.REC_PRN * 4)
                    if int(prn_f) == 4:
                        off = base + telem.REC_PHI0 * 4
                        (v,) = struct.unpack_from("<f", b, off)
                        struct.pack_into("<f", b, off, v + math.pi)
        lost.append((bytes(b), chain))
    lo = _xc(lost)
    if lo is None:
        return 1
    for c in sorted({c for c, _ in co}):
        if not lo[(c, 4)] < co[(c, 4)] - 0.2:
            print(
                "FAIL -- %s PRN 4: one sender lost its reference and xcoh did not fall "
                "(%.3f -> %.3f)" % (c, co[(c, 4)], lo[(c, 4)])
            )
            return 1
        others = [
            abs(lo[(c, p)] - co[(c, p)])
            for (cc, p) in co
            if cc == c
            and p != 4
            and co[(cc, p)] is not None
            and lo.get((cc, p)) is not None
        ]
        if max(others) > 1e-9:
            print(
                "FAIL -- %s: moving PRN 4's phi0 on one sender moved another PRN's xcoh "
                "by %.3g" % (c, max(others))
            )
            return 1
    print(
        "XCOH PASS -- co-phased signal PRNs >= %.3f, pure noise <= %.3f in |xcoh|; one "
        "sender's reference loss on PRN 4 read %.2f -> %.2f and touched no other PRN."
        % (lo_sig, hi_nz, co[("gps_l5", 4)], lo[("gps_l5", 4)])
    )

    # ---- The per-record series (the served C/N0's input) --------------------------------
    pr = python_recs(frames, args.n_win)
    bad_r = compare_recs(pr, cpp)
    if bad_r:
        print("FAIL -- %d per-record disagreement(s):" % len(bad_r))
        for sev, msg in bad_r[:12]:
            print("  %-12s %s" % (sev, msg))
        return 1
    n_rr = sum(len(v) for ch in pr.values() for v in ch.values())
    print(
        "REC-SERIES PASS -- %d (window, slot, PRN) records agree, e/p/l to 1e-9 and "
        "n_inst/n_chan exactly." % n_rr
    )

    # ---- The shipping configuration: taps_win != n_win ---------------------------------
    # Production runs the loop on 2 windows and serves the broker 32. A gate run only at
    # taps_win == n_win would never touch the split, so both halves are asserted here:
    #   (a) the served taps deepen to the requested window count, and
    #   (b) the loop does not move. That is the risk of sharing one ring: if aggregate()
    #       walked the deeper slice, the discriminator would average taps_win windows where
    #       the policy asks for n_win, and every trim step would be formed on stale sky.
    deep = 2 * args.n_win
    c_deep = cpp_arm(args.exe, path, args.n_win, args.min_instances, taps_win=deep)
    if compare(py, c_deep):
        print(
            "FAIL -- deepening taps_win CHANGED the discriminator. aggregate() must keep "
            "walking only the newest n_win of the ring."
        )
        return 1
    bad_d = compare_taps(python_taps(frames, deep, args.min_instances), c_deep)
    if bad_d:
        print("FAIL -- %d disagreement(s) at taps_win=%d:" % (len(bad_d), deep))
        for sev, msg in bad_d[:12]:
            print("  %-11s %s" % (sev, msg))
        return 1
    if c_deep.get("taps_win") != deep:
        print(
            "FAIL -- asked for taps_win %d, tool reports %s"
            % (deep, c_deep.get("taps_win"))
        )
        return 1
    print(
        "SPLIT-RING PASS -- taps deepen to %d windows and the %d-window loop is unmoved."
        % (deep, args.n_win)
    )

    # ---- The integrator -----------------------------------------------------------------
    # Two legs, kept separate because a single end-to-end comparison would let a fold error
    # and a recurrence error cancel:
    #   (a) the C++ disc at every window close matches Python's on the same window set;
    #   (b) Python's dll_integrate, driven by that disc series, reproduces the C++ trim series.
    # (b) alone would be circular without (a); (a) extends the discriminator leg above from
    # the last window to every window. Both run at GATE_NOBODY: the series also records
    # leak-only updates with the last stepped disc, which (b) would replay as steps.
    ARM = list(
        SIGNAL_PRNS
    )  # see SIGNAL_PRNS: the near-noise PRNs are never armed, only the real signals
    cpp2 = cpp_arm(args.exe, path, args.n_win, args.min_instances, arm=ARM)
    series = cpp2.get("series", {}).get("gps_l5", {})
    if not series:
        print(
            "FAIL -- the C++ integrator stepped on nothing. It was armed on %s." % ARM
        )
        return 1
    wins_seen = sorted({int(r[0]) for rows in series.values() for r in rows})
    pyd = python_disc_series(
        frames, "gps_l5", wins_seen, args.n_win, args.min_instances
    )

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
                bad_i.append(
                    "PRN %d win %d: C++ stepped, Python formed no discriminator"
                    % (prn, win)
                )
                continue
            if abs(ref - disc) > 1e-9 * max(1.0, abs(ref)):
                bad_i.append(
                    "PRN %d win %d disc: py %.12g vs cpp %.12g" % (prn, win, ref, disc)
                )
            n_disc += 1
            # (b) the recurrence, driven by the same disc. Loop constants only: p_floor_abs
            # is the C++ integrator's window gate, and dll_integrate takes no such argument.
            trim = combdll.dll_integrate(trim, disc, **LOOP_POLICY)
            if abs(trim - cpp_trim) > 1e-12 * max(1.0, abs(trim)):
                bad_i.append(
                    "PRN %d win %d trim: py %.12g vs cpp %.12g"
                    % (prn, win, trim, cpp_trim)
                )
            n_trim += 1
    if bad_i:
        print("\nFAIL -- integrator: %d disagreement(s):" % len(bad_i))
        for m in bad_i[:20]:
            print("  %s" % m)
        return 1

    # And check that it moved: an integrator that returned 0 every step would pass every
    # comparison above, so some final trim must be non-trivial. The leak ceiling is printed
    # beside the largest trim for reference.
    finals = {int(p): rows[-1][2] for p, rows in series.items() if rows}
    ceiling = DEFAULT_POLICY["gain"] * 0.25 / DEFAULT_POLICY["leak"]
    if max(abs(v) for v in finals.values()) < 1e-3:
        print("FAIL -- every trim is ~0. The integrator ran but did not integrate.")
        return 1
    print(
        "INTEGRATOR PASS -- %d window discriminators and %d integrator steps agree "
        "(%d PRNs, %d steps each)."
        % (n_disc, n_trim, len(series), len(next(iter(series.values()))))
    )
    print(
        "  final trims: %s"
        % ", ".join("PRN %d %+.4f" % (p, v) for p, v in sorted(finals.items()))
    )
    # ---- The trim store: a gather restart must not cost the fleet its pull-in -----------
    # GnssFleetTrim holds its trims in process; without the store a restart leaves every
    # tracker off-peak by its standing trim. This asserts the save/adopt path including its
    # refusals: a store adopted when it should not be is worse than none, because it commands
    # a code step nothing verified.
    snap = cpp2.get("trim_snapshot", {}).get("chains", {})
    if not snap.get("gps_l5"):
        print("FAIL -- the armed run produced no trim snapshot to persist.")
        return 1
    store = os.path.join(tmp, "trims.json")
    with open(store, "w") as fh:
        json.dump({"saved_unix": 0.0, "chains": snap}, fh)

    # (a) Adopted when the broker arms the PRN, and the loop resumes from the stored value.
    c_res = cpp_arm(
        args.exe, path, args.n_win, args.min_instances, arm=ARM, trim_in=store
    )
    n_stored = sum(len(v) for v in snap.values())  # every chain, not just gps_l5
    if c_res.get("trim_adopted", 0) != n_stored:
        print(
            "FAIL -- offered %d stored trim(s) across %d chain(s), adopted %d."
            % (n_stored, len(snap), c_res.get("trim_adopted"))
        )
        return 1
    # The restored run must not reproduce the cold run: it starts from the stored trims.
    cold = cpp2.get("series", {}).get("gps_l5", {})
    warm = c_res.get("series", {}).get("gps_l5", {})
    first_cold = {p: v[0][2] for p, v in cold.items() if v}
    first_warm = {p: v[0][2] for p, v in warm.items() if v}
    if not first_warm or first_cold == first_warm:
        print(
            "FAIL -- the restored run's FIRST trim equals the cold run's. The store was "
            "loaded and had no effect, which is persistence that measures as working."
        )
        return 1

    # (b) Refused when nothing armed that PRN: adoption is gated on arming, not done at load,
    # so policy stays with the broker.
    c_noarm = cpp_arm(args.exe, path, args.n_win, args.min_instances, trim_in=store)
    if c_noarm.get("trim_adopted", 0) != 0:
        print(
            "FAIL -- adopted %d stored trim(s) with NOTHING armed. A restored trim is a "
            "proposal; arming is the acceptance." % c_noarm.get("trim_adopted")
        )
        return 1

    # (c) Refused when the value exceeds the clamp. A store is a file on disk and can hold
    # anything; the clamp is the contract.
    bad = os.path.join(tmp, "trims_bad.json")
    with open(bad, "w") as fh:
        json.dump(
            {"saved_unix": 0.0, "chains": {"gps_l5": {str(p): 1e9 for p in ARM}}}, fh
        )
    c_bad = cpp_arm(
        args.exe, path, args.n_win, args.min_instances, arm=ARM, trim_in=bad
    )
    if c_bad.get("trim_adopted", 0) != 0:
        print(
            "FAIL -- adopted a trim beyond the clamp (%d of them)."
            % c_bad.get("trim_adopted")
        )
        return 1
    print(
        "TRIM-STORE PASS -- %d trim(s) across %d chain(s) persist and are adopted AT ARMING "
        "(max |trim| %.3f chips), refused unarmed (0 adopted) and refused beyond the clamp "
        "(0 adopted)."
        % (n_stored, len(snap), max(abs(v) for c in snap.values() for v in c.values()))
    )

    print(
        "  leak ceiling gain*0.25/leak = %.3f chips; max |trim| here %.3f"
        % (ceiling, max(abs(v) for v in finals.values()))
    )

    # ---- The absolute window gate (TrimPolicy::p_floor_abs) ----------------------------
    # integrate() steps an armed PRN only in windows whose p_pow reaches the broker's
    # probe-anchored floor; other windows are leak-only. The floor is absolute: one relative
    # to the other PRNs' powers would compare satellites with each other, not with noise. A
    # floor between two armed satellites' powers must turn the weaker one fully leak-only and
    # leave the stronger one's skip count and trim series identical to the GATE_NOBODY
    # baseline: the floor changes who is gated, never the arithmetic.
    rows_a = {
        int(k): v
        for k, v in cpp2.get("chains", {}).get("gps_l5", {}).get("prns", {}).items()
    }
    p_of = {p: rows_a[p]["p_pow"] for p in ARM if p in rows_a}
    if len(p_of) < 2:
        print(
            "FAIL -- the abs-floor case needs two armed PRNs with rows; got %s" % p_of
        )
        return 1
    lo_prn = min(p_of, key=p_of.get)
    hi_prn = max(p_of, key=p_of.get)
    floor_mid = (p_of[lo_prn] + p_of[hi_prn]) / 2.0
    pol_mid = dict(DEFAULT_POLICY, p_floor_abs=floor_mid)
    c_mid = cpp_arm(
        args.exe, path, args.n_win, args.min_instances, arm=ARM, pol=pol_mid
    )
    cen_a = cpp2.get("census", {}).get("gps_l5", {})
    cen_m = c_mid.get("census", {}).get("gps_l5", {})
    lo_a, lo_m = cen_a.get(str(lo_prn), {}), cen_m.get(str(lo_prn), {})
    hi_m = cen_m.get(str(hi_prn), {})
    if not (
        lo_m.get("skipped", 0) > lo_a.get("skipped", 0) and lo_m.get("steps", 1) == 0
    ):
        print(
            "FAIL -- abs floor %.3g should turn PRN %d (p %.3g) fully leak-only; census "
            "went %s -> %s" % (floor_mid, lo_prn, p_of[lo_prn], lo_a, lo_m)
        )
        return 1
    if hi_m.get("skipped", 0) != cen_a.get(str(hi_prn), {}).get("skipped", 0):
        print(
            "FAIL -- abs floor %.3g touched PRN %d (p %.3g), which sits ABOVE it."
            % (floor_mid, hi_prn, p_of[hi_prn])
        )
        return 1
    if c_mid.get("series", {}).get("gps_l5", {}).get(str(hi_prn)) != cpp2.get(
        "series", {}
    ).get("gps_l5", {}).get(str(hi_prn)):
        print(
            "FAIL -- the above-floor PRN %d's trim SERIES moved under the abs floor. The "
            "flag must change only who is gated, never the arithmetic." % hi_prn
        )
        return 1
    # No floor means refuse, never "no gate": p_floor_abs <= 0 must leak every armed window.
    # Ungated, an off-peak discriminator flips sign every window and random-walks the trim to
    # the clamp. This leg fails if 0 is ever treated as no gate.
    pol_none = dict(DEFAULT_POLICY, p_floor_abs=0.0)
    c_none = cpp_arm(
        args.exe, path, args.n_win, args.min_instances, arm=ARM, pol=pol_none
    )
    cen_n = c_none.get("census", {}).get("gps_l5", {})
    stepped = {p: cen_n.get(str(p), {}).get("steps", 0) for p in ARM}
    if any(v for v in stepped.values()):
        print(
            "FAIL -- with NO absolute floor the loop must refuse (leak-only, steps 0) and "
            "instead it stepped: %s. A missing noise reference is not a licence to gate on "
            "nothing." % stepped
        )
        return 1
    if not all(cen_n.get(str(p), {}).get("skipped", 0) > 0 for p in ARM):
        print(
            "FAIL -- refusal must LEAK every armed PRN, not silently drop them: %s"
            % cen_n
        )
        return 1
    print(
        "ABS-FLOOR PASS -- floor %.3g: PRN %d (below) fully leak-only, PRN %d (above) "
        "byte-identical; floor 0 REFUSES every armed PRN (steps 0, all leak-only)."
        % (floor_mid, lo_prn, hi_prn)
    )

    if args.self_test:
        # The gate must be able to fail. Nudge one channel's early tap in one record
        # (x1.01 + 1.0) and require the comparison to notice; if it does not, the comparison
        # is not comparing.
        #
        # The poked frame must be one the answer depends on: a perturbation outside the
        # averaging ring is invisible by design (with n_win 4 and 12 windows the ring holds
        # windows 7..10). With --no-flush the newest window stays open, so window W-2 is the
        # newest closed one and is in the ring for any n_win >= 1.
        n_per_win = 2 * len(INSTANCES)  # two chains
        poke_at = (args.windows - 2) * n_per_win
        buf = bytearray(frames[poke_at][0])
        # Poke a row whose PRN contributes, found by PRN rather than by a fixed index: the
        # row order is the fixture's shuffle (row compaction), and a fixed row could land on
        # PRN 11, which min_instances excludes, leaving the poke inert. The stride is the poked
        # sender's own, off its header (n_row_total), never a constant: senders here carry
        # different comb widths.
        _row_total = telem._HDR.unpack_from(buf, 0)[15]
        _row = None
        for _r in range(N_PRN):
            _o = telem._HDR_BYTES + _r * _row_total * 4
            if int(struct.unpack_from("<f", buf, _o)[0] + 0.5) in SIGNAL_PRNS:
                _row = _r
                break
        if _row is None:
            raise SystemExit(
                "self-test: no SIGNAL_PRNS row in the poked frame -- the fixture "
                "changed shape and the poke would be inert"
            )
        off = (
            telem._HDR_BYTES
            + _row * _row_total * 4
            + telem._ROW_FLOATS * 4
            + telem.CHAN_E_RE * 4
        )
        (v,) = struct.unpack_from("<f", buf, off)
        struct.pack_into("<f", buf, off, v * 1.01 + 1.0)
        poked = (
            frames[:poke_at]
            + [(bytes(buf), frames[poke_at][1])]
            + frames[poke_at + 1 :]
        )
        p2 = os.path.join(tmp, "frames_poked.bin")
        write_stream(p2, poked)
        bad2 = compare(
            python_arm(poked, args.n_win, args.min_instances),
            cpp_arm(args.exe, p2, args.n_win, args.min_instances),
        )
        # Both arms see the same poked bytes, so they must still agree with each other; the
        # check below shows the comparison is sensitive at all by requiring the poked answer to
        # differ from the clean one.
        if bad2:
            print(
                "SELF-TEST FAIL: the arms disagree on the perturbed fixture: %s"
                % bad2[:3]
            )
            return 1
        moved = compare(py, cpp_arm(args.exe, p2, args.n_win, args.min_instances))
        if not moved:
            print(
                "SELF-TEST FAIL: a 1%% perturbation of one channel (window %d, the newest "
                "closed one) changed NOTHING -- the comparison, the fixture, or the fold is "
                "not doing what it claims." % (args.windows - 2)
            )
            return 1
        print(
            "SELF-TEST PASS -- the arms still agree on perturbed bytes, and the comparison "
            "sees the perturbation (%d field(s) moved)." % len(moved)
        )

        # And the rotation must matter. Move one sender's phi0 on that same row by 1 rad
        # without touching its comb: that sender's partial sum now lands off the common
        # reference and the lobe power must change. If it does not, the arms agree on a sum
        # that ignores REC_PHI0: coherent per sender, not per lobe.
        buf = bytearray(frames[poke_at][0])
        off = telem._HDR_BYTES + _row * _row_total * 4 + telem.REC_PHI0 * 4
        (v,) = struct.unpack_from("<f", buf, off)
        struct.pack_into("<f", buf, off, v + 1.0)
        poked = (
            frames[:poke_at]
            + [(bytes(buf), frames[poke_at][1])]
            + frames[poke_at + 1 :]
        )
        p4 = os.path.join(tmp, "frames_phi0.bin")
        write_stream(p4, poked)
        c4 = cpp_arm(args.exe, p4, args.n_win, args.min_instances)
        if compare(python_arm(poked, args.n_win, args.min_instances), c4):
            print(
                "SELF-TEST FAIL: the arms disagree once one sender's phi0 moves -- they do "
                "not rotate the same way."
            )
            return 1
        moved = compare(py, c4)
        if not moved:
            print(
                "SELF-TEST FAIL: moving one sender's REC_PHI0 by 1 rad changed NOTHING. The "
                "senders are being added as powers, not as rotated complex sums."
            )
            return 1
        print(
            "PHI0 SELF-TEST PASS -- one sender's reference moved 1 rad, both arms agree and "
            "the lobe power moved (%d field(s))." % len(moved)
        )
    if not args.keep:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
    else:
        print("kept in %s" % tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
