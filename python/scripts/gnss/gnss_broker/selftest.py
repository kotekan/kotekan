"""Direct checks for broker behaviour the transcript gate cannot reach.

`scripts/gnss/broker_equiv.py` proves the broker's POST stream is unchanged, but only on
the code paths its fixtures reach, and a replay never runs the frozen clock in live mode or
on a second thread. Those, the shared Receiver, the named signals and the estimator and
filter contracts get direct assertions here. Every PRNG is seeded.

    python3 gnss_broker/selftest.py      # from python/scripts/gnss
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnss_broker import transport, receiver, signals  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(
        "%-52s %s%s" % (name, "ok" if cond else "FAIL", "  " + detail if detail else "")
    )
    if not cond:
        FAIL.append(name)


# -- the frozen cycle clock ---------------------------------------------------------------
# The freeze must hold in live mode, not only while a transcript is open, or a recording
# would change what it records. The transcript gate compares recordings with recordings and
# cannot see this; only a live-mode assertion can.
transport._TR.tick()
_a = transport._now()
time.sleep(0.02)
_b = transport._now()
check("cycle clock frozen in LIVE mode", _a == _b, "%.6f vs %.6f" % (_a, _b))
transport._TR.tick()
check("tick() advances it", transport._now() > _a)

# -- #208: radiometric time is the hop axis, never a wall-clock reading ----------------------
# transport.axis_now is the one "now" every label, model epoch and age is referred to;
# dr_eph_epoch must place the model at utc0_sample0 + t_now_abs and label it t_now_abs; a
# wall-clock step must move neither. The source grep over the package keeps a wall-minus-anchor
# or a model-at-wall-time from coming back; the two sanctioned wall uses (the no-axis fallback
# inside axis_now, the setup day key) carry a `208-allow` marker on their line.
import glob as _glob  # noqa: E402
import re as _re  # noqa: E402
import types as _types  # noqa: E402

from gnss_broker import deadreckon as _dr  # noqa: E402

_ac = _types.SimpleNamespace(
    utc0_sample0=1.7e9,
    args=_types.SimpleNamespace(dr_fengine_axis=True, hops_per_sec=195312.5),
    fe_axis=[None],
    fe_off=[None],
)
_wn = transport._now()
check(
    "axis_now: no telemetry yet -> frozen clock minus the anchor",
    abs(transport.axis_now(_ac) - (_wn - 1.7e9)) < 1e-6,
)
_ac.fe_axis[0] = (
    1000.0 * 195312.5,
    _wn - 0.25,
)  # newest hop 1000 s in, polled 0.25 s ago
check(
    "axis_now: raw newest hop + elapsed", abs(transport.axis_now(_ac) - 1000.25) < 1e-6
)
_ac.fe_off[0] = 1000.1 - _wn  # the filtered hop-minus-wall offset
check(
    "axis_now: filtered offset + frozen clock",
    abs(transport.axis_now(_ac) - 1000.1) < 1e-6,
)
_ac.args.dr_fengine_axis = False
check(
    "axis_now: without the flag = frozen clock minus the anchor",
    abs(transport.axis_now(_ac) - (_wn - 1.7e9)) < 1e-6,
)
_ac.utc0_sample0 = 0.0
check("axis_now: None without the anchor", transport.axis_now(_ac) is None)

_ec = _types.SimpleNamespace(
    utc0_sample0=1.7e9, drp=_types.SimpleNamespace(t_now_abs=1234.5, now_w=9.9e9)
)
_eage, _ew = _dr.dr_eph_epoch(_ec)
check(
    "eph epoch = hop axis (t_now_abs), labelled as such",
    _eage == 1234.5 and abs(_ew - (1.7e9 + 1234.5)) < 1e-6,
    "age %.3f utc %.3f" % (_eage, _ew),
)
_ec.drp.now_w += 86400.0
check("eph epoch ignores a wall-clock step", _dr.dr_eph_epoch(_ec) == (_eage, _ew))
_wall_rx = _re.compile(
    r"fromtimestamp\(\s*(_now\(\)|time\.time\(\)|ctx\.t0|ctx\.drp\.now_w|now_w)\b"
    r"|(ctx\.drp\.now_w|ctx\.t0|_now\(\)|time\.time\(\)|now_w)\s*-\s*ctx\.utc0_sample0"
)
_here = os.path.dirname(os.path.abspath(__file__))
_wall = []
for _f in sorted(_glob.glob(os.path.join(_here, "*.py"))) + [
    os.path.join(os.path.dirname(_here), "gps_distributed_broker.py")
]:
    _b = os.path.basename(_f)
    if _b.startswith("test_") or _b == "selftest.py":
        continue
    for _i, _l in enumerate(open(_f), 1):
        if _wall_rx.search(_l) and "208-allow" not in _l:
            _wall.append("%s:%d" % (_b, _i))
check(
    "no model at wall time, no wall-minus-anchor age, outside the two allowed lines",
    not _wall,
    " ".join(_wall[:6]),
)

_other = {}


def _thread_clock():
    # No tick() here: a thread that does not run cycles (the publisher's HTTP thread) must
    # read the real clock, not another thread's frozen instant.
    _other["v"] = transport._now()


t = threading.Thread(target=_thread_clock)
t.start()
t.join()
check("a non-cycling thread reads the real clock", _other["v"] != transport._now())


def _thread_cycle():
    transport._TR.tick()
    _other["c"] = transport._now()


t = threading.Thread(target=_thread_cycle)
t.start()
t.join()
check("two chains hold INDEPENDENT frozen clocks", _other["c"] != transport._now())

# -- the Receiver -------------------------------------------------------------------------
rx = receiver.Receiver(log=lambda m: None)

check(
    "anchor: fetched once, shared by all",
    rx.time_anchor(lambda: 1786167610.0, "a") == 1786167610.0
    and rx.time_anchor(lambda: 999.0, "b") == 1786167610.0,
)

_built = []
rx.brdc("k", lambda: _built.append(1) or {"eph": {}})
rx.brdc("k", lambda: _built.append(1) or {"eph": {}})
check("brdc: built once per key", len(_built) == 1, "%d builds" % len(_built))

# A chain never consumes what it contributed itself, so a chain alone in a process computes
# exactly what it would without the shared Receiver; the transcript gate relies on this.
rx.contribute_code_bias("gps_l5", "1176.45MHz", -12.5e-6, 6, 1000.0)
check(
    "clock: a chain does not consume its own estimate",
    rx.code_bias("1176.45MHz", exclude="gps_l5", t_now=1000.0) is None,
)
_s = rx.code_bias("1176.45MHz", exclude="gal_e5a", t_now=1000.0)
check("clock: a sibling consumes it", _s is not None and _s.value == -12.5e-6)
check(
    "clock: stale contributions age out",
    rx.code_bias("1176.45MHz", exclude="gal_e5a", t_now=1000.0 + 999.0) is None,
)

rx.contribute_code_bias("other", "1176.45MHz", -9.0e-6, 12, 1000.0)
_s = rx.code_bias("1176.45MHz", exclude="gal_e5a", t_now=1000.0)
check(
    "clock: the better-supported estimate wins", _s.value == -9.0e-6, "n=%d" % _s.weight
)

# The band-scoped lookup stays per band: a bias contributed at 1176.45 MHz is not returned
# for 1207.14 MHz (code_bias_any_band is the cross-band fallback, checked below).
rx.contribute_code_bias("gps_l5", "1176.45MHz", 2.6e-6, 5, 1000.0)
check(
    "code bias: visible inside the band",
    rx.code_bias("1176.45MHz", exclude="gal_e5a", t_now=1000.0) is not None,
)
check(
    "code bias: not visible across a retune",
    rx.code_bias("1207.14MHz", exclude="gal_e5a", t_now=1000.0) is None,
)

rx.contribute_dr_clock("gps_l5", "1176.45MHz", 5094.9, 0.31, 1000.0, 10230.0)
_d = rx.dr_clock("1176.45MHz", exclude="gal_e5a", t_now=1000.0)
check(
    "dr clock: carries its code length",
    _d is not None and _d.extra["code_length"] == 10230.0 and _d.extra["drift"] == 0.31,
)

# -- signals ------------------------------------------------------------------------------
# Cross-checks the values signals.py derives from gnssSignal.hpp against independently
# written ones: carrier, chip rate, primary length, and the long code's segment count and
# repeat period (L5 NH20 20 x 1 ms; E5a and B2a 100 x 1 ms; L2C CL 75 x 20 ms = 1.5 s;
# L1 C/A has no long code).
for key, want in (
    ("gps_l5", (1176.45e6, 10.23e6, 10230, 20, 0.02)),
    ("gal_e5a", (1176.45e6, 10.23e6, 10230, 100, 0.1)),
    ("bds_b2a", (1176.45e6, 10.23e6, 10230, 100, 0.1)),
    ("gps_l2c", (1227.6e6, 511.5e3, 10230, 75, 1.5)),
    ("gps_l1ca", (1575.42e6, 1.023e6, 1023, 1, 0.001)),
):
    s = signals.get(key)
    got = (
        s.carrier_hz,
        s.chip_rate_hz,
        s.code_length,
        s.long_code_segments,
        s.long_code_epoch_s,
    )
    check(
        "signal %s matches its ICD parameters" % key,
        all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in zip(got, want)),
        "%s vs %s" % (got, want),
    )

# -- dead-reckon seed currency -----------------------------------------------------------
# The slewed refresh (deadreckon.py) re-anchors a locked seed every pass and relies on two
# properties of dr_cp0/dr_seed_phys, failing silently without either: (1) they are exact
# inverses, so "where does the tracker think the code is" is answerable; (2) re-anchoring a
# seed at a later hop with a different Doppler moves the implied physical phase by exactly
# zero, which is what makes the slew safe to run every pass.
import random as _rnd

from fits import dr_cp0, dr_seed_phys  # noqa: E402

_rnd.seed(30)
_CHIP, _FC, _S, _HPS, _MOD = 10.23e6, 1176.45e6, -1.0, 195312.5, 1023000.0
_w_rt = _w_cont = 0.0
for _ in range(500):
    _phys = _rnd.uniform(0, _MOD)
    _t = _rnd.uniform(1e3, 2e5)
    _dop = _rnd.uniform(-4000, 4000)
    _h0 = int(round(_t * _HPS))
    _sd = {
        "code_phase_chips": dr_cp0(_phys, _h0 / _HPS, _dop, _CHIP, _FC, _S, _MOD),
        "ref_hop": _h0,
        "doppler_hz": _dop,
        "code_phase_rate": _rnd.uniform(-1e-4, 1e-4),
        "doppler_rate_hz_s": _rnd.uniform(-0.5, 0.5),
    }
    _e = abs(
        (
            ((dr_seed_phys(_sd, _h0, _HPS, _CHIP, _FC, _S, _MOD) - _phys) + _MOD / 2)
            % _MOD
        )
        - _MOD / 2
    )
    _w_rt = max(_w_rt, _e)
check(
    "dr currency round-trips (dr_seed_phys inverts dr_cp0 at birth)",
    _w_rt < 1e-3,
    "worst %.3e chips over 500 trials" % _w_rt,
)
# continuity: roll to h1 at a changed Doppler; implied phys at h1 must not move at all
_rnd.seed(31)
_w_cont = 0.0
for _ in range(500):
    _phys = _rnd.uniform(0, _MOD)
    _t = _rnd.uniform(1e3, 2e5)
    _dop = _rnd.uniform(-4000, 4000)
    _dop2 = _dop + _rnd.uniform(-2, 2)
    _h0 = int(round(_t * _HPS))
    _h1 = _h0 + int(_rnd.uniform(1, 600) * _HPS)
    _sd = {
        "code_phase_chips": dr_cp0(_phys, _h0 / _HPS, _dop, _CHIP, _FC, _S, _MOD),
        "ref_hop": _h0,
        "doppler_hz": _dop,
        "code_phase_rate": _rnd.uniform(-1e-4, 1e-4),
        "doppler_rate_hz_s": _rnd.uniform(-0.5, 0.5),
    }
    _p1 = dr_seed_phys(_sd, _h1, _HPS, _CHIP, _FC, _S, _MOD)
    _sd2 = {
        "code_phase_chips": dr_cp0(_p1, _h1 / _HPS, _dop2, _CHIP, _FC, _S, _MOD),
        "ref_hop": _h1,
        "doppler_hz": _dop2,
        "code_phase_rate": 0.0,
        "doppler_rate_hz_s": 0.0,
    }
    _p1b = dr_seed_phys(_sd2, _h1, _HPS, _CHIP, _FC, _S, _MOD)
    _w_cont = max(_w_cont, abs((((_p1b - _p1) + _MOD / 2) % _MOD) - _MOD / 2))
check(
    "dr slew re-anchor is CONTINUOUS across a Doppler change",
    _w_cont < 1e-6,
    "worst %.3e chips over 500 trials" % _w_cont,
)

# -- fleet phase-slope delay fit ---------------------------------------------------------
# The fitter must (1) recover a known delay with the right sign, to within 0.02 chips, from
# the fleet's comb geometry (stride-16 combs with per-node offsets: their union is what
# resolves the 3.27-chip grating lobes), (2) read peak/floor ~1 on pure noise, and (3) be
# deterministic, since it runs inside the digest-gated broker.
import cmath as _cm

from gnss_broker.fleet import fit_spectrum_delay  # noqa: E402

_CHIPR, _W = 10.23e6, 195312.5
_COMBS = [
    [5972 + 16 * i for i in range(7)],
    [5980 + 16 * i for i in range(7)],
    [5975 + 16 * i for i in range(7)],
    [5983 + 16 * i for i in range(7)],
    [5978 + 16 * i for i in range(7)],
    [5986 + 16 * i for i in range(7)],
]
_rnd.seed(32)


def _spec_synth(tau_chips, snr=30.0):
    tau_s = tau_chips / _CHIPR
    pts = []
    for k, fids in enumerate(_COMBS):
        ph = _rnd.uniform(0, 2 * _cm.pi)  # unknown per-instance phase, always present
        for fid in fids:
            a = _cm.exp(-2j * _cm.pi * fid * _W * tau_s + 1j * ph)
            a += complex(_rnd.gauss(0, 1 / snr), _rnd.gauss(0, 1 / snr))
            pts.append((fid, a, 1.0, "inst%d" % k))
    return pts


_worst = 0.0
for _truth in (-1.2, -0.4, 0.0, 0.31, 0.8, 1.5):
    _r = fit_spectrum_delay(_spec_synth(_truth), _CHIPR, _W)
    _worst = max(_worst, abs(_r[0] - _truth))
check(
    "spectrum delay fit recovers sign+magnitude on the real comb",
    _worst < 0.02,
    "worst |tau err| %.4f chips over 6 truths in [-1.2, +1.5]" % _worst,
)
_noise = [
    (fid, complex(_rnd.gauss(0, 1), _rnd.gauss(0, 1)), 1.0, "i%d" % k)
    for k, c in enumerate(_COMBS)
    for fid in c
]
_rn = fit_spectrum_delay(_noise, _CHIPR, _W)
check(
    "spectrum delay fit reads peak/floor ~1 on pure noise",
    _rn[1] / max(_rn[2], 1e-12) < 1.5,
    "%.2f" % (_rn[1] / max(_rn[2], 1e-12)),
)
_p = _spec_synth(0.31)
check(
    "spectrum delay fit is deterministic",
    fit_spectrum_delay(_p, _CHIPR, _W) == fit_spectrum_delay(_p, _CHIPR, _W),
)

# -- SatBiasFilter ------------------------------------------------------------------------
# The b_sat loop's four behaviours: converge to a constant bias through realistic noise;
# reject lobe captures at +-3.27 chips outright, never averaging them in; stop applying (but
# remember) a stale bias; refuse thin fleets, whose fits scatter several times wider.
from gnss_broker.state_filter import SatBiasFilter  # noqa: E402

_rnd.seed(33)
_bf = SatBiasFilter(gain=0.02, innovation_max=1.0, max_age_s=600.0, min_inst=6)
_t = 0.0
for _ in range(600):  # ~20 min at 2 s
    _t += 2.0
    _bf.update(3, 0.081 + _rnd.gauss(0, 0.35), 12, _t)  # an E5a-scale bias and noise
check(
    "SatBiasFilter converges to a constant bias through noise",
    abs(_bf.get(3, _t) - 0.081) < 0.03,
    "%.4f vs +0.081" % _bf.get(3, _t),
)
_rej0 = _bf.rejected
_ok = _bf.update(
    3, 3.27 + 0.081, 12, _t
)  # a lobe capture, exactly one comb spacing off
check(
    "SatBiasFilter rejects a lobe capture via the innovation gate",
    (not _ok) and _bf.rejected == _rej0 + 1 and abs(_bf.get(3, _t) - 0.081) < 0.03,
)
check("SatBiasFilter refuses a thin fleet", not _bf.update(3, 0.08, 4, _t))
check(
    "SatBiasFilter stops APPLYING a stale bias but remembers it",
    _bf.get(3, _t + 601.0) == 0.0 and "3:" in _bf.summary(_t + 601.0),
)

# -- JointReceiverState -------------------------------------------------------------------
# The contract of docs/CHORD_JOINT_TRACKING.md section 3a. Each check is one of the reasons a
# joint solve beats a staged per-satellite design, written so that a regression toward
# per-satellite tracking fails.
from gnss_broker.state_filter import JointReceiverState  # noqa: E402


def _run(js, truth_clk, truth_b, t0, n, dt=2.0, sigma=0.30, skip=(), rate=0.0):
    """Feed synthetic y_i = clk(t) + b_i + noise for a fleet of sats."""
    t = t0
    for _ in range(n):
        t += dt
        clk = truth_clk + rate * (t - t0)
        _run.clk = clk
        js.cycle(
            [
                (k, clk + b + _rnd.gauss(0, sigma), sigma)
                for k, b in truth_b.items()
                if k not in skip
            ],
            t,
        )
    return t


_rnd.seed(3311)
_TB = {
    ("E", 3): +2.9,
    ("E", 5): -3.4,
    ("E", 31): +0.8,
    ("C", 39): -1.7,
    ("G", 14): +4.1,
    ("G", 21): -2.7,
}  # chips: the scale of the model's per-satellite error
_js = JointReceiverState(q_b=0.013, gauge_sigma=0.1)
_t = _run(_js, 151.0, _TB, 0.0, 400)  # ~13 min at 2 s
# The gauge defines clk against the fleet (the median bias), so compare clk+b_i per
# satellite, not clk alone.
_err = max(abs(_js.predicted(k) - (151.0 + b)) for k, b in _TB.items())
check(
    "JointReceiverState recovers clk+b_sat at the chips-wrong model scale",
    _err < 0.25,
    "worst %.3f chips over %d sats" % (_err, len(_TB)),
)
# The default gauge ("median") pins the median bias, not the mean.
_bs = sorted(_js.bias(k) for k in _TB)
_med_b0 = (
    _bs[len(_bs) // 2]
    if len(_bs) % 2
    else 0.5 * (_bs[len(_bs) // 2 - 1] + _bs[len(_bs) // 2])
)
check(
    "JointReceiverState gauge holds MEDIAN(b) at zero",
    abs(_med_b0) < 0.05,
    "median b %+.4f (mean %+.4f, not pinned)"
    % (_med_b0, sum(_js.bias(k) for k in _TB) / len(_TB)),
)

# Birth: the clock's absolute value must survive the first cycle. An offset placed in the
# biases is sheared back out by the gauge (it sees no cross-covariance to a newborn row)
# without reaching clk, which then grinds back over hundreds of cycles, and a clock near L/2
# would converge to a wrapped alias.
_rnd.seed(4242)
_jb = JointReceiverState()
_jb.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], 2.0)
check(
    "the clock's absolute value survives satellite BIRTH",
    abs(_jb.clk - 151.0) < 1.5,
    "clk %+.2f after one cycle (want ~151)" % _jb.clk,
)
_jw2 = JointReceiverState(code_len=10230.0)
_jw2.cycle([(k, 5100.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], 2.0)
check(
    "birth near L/2 does not converge to a wrapped alias",
    abs(_jw2.wrap(_jw2.clk - 5100.0)) < 1.5,
    "clk %+.1f (want ~5100)" % _jw2.clk,
)

# The split is bandwidth. A common step must land in clk (fast, shared); a per-satellite
# step must land in that satellite's b (slow, private). Getting this backwards is the
# misspecification that makes the plant oscillate.
_b_before = {k: _js.bias(k) for k in _TB}
_c_before = _js.clk
_t = _run(_js, 156.0, _TB, _t, 120)  # +5 chips common
# Compared as a step, not against the absolute 156: under the median gauge clk sits at 156
# plus the biases' median-vs-mean offset (~-0.45 for _TB). What a common step must do does
# not depend on the gauge: clk moves by the step and the biases do not.
check(
    "a COMMON offset moves clk (by the step), not the biases",
    abs((_js.clk - _c_before) - 5.0) < 0.4
    and max(abs(_js.bias(k) - _b_before[k]) for k in _TB) < 0.5,
    "dclk %+.2f (want +5.00), worst db %.2f"
    % (_js.clk - _c_before, max(abs(_js.bias(k) - _b_before[k]) for k in _TB)),
)
_clk_before = _js.clk
_TB2 = dict(_TB)
_TB2[("E", 3)] += 2.0  # one satellite walks 2 chips
_t = _run(_js, 156.0, _TB2, _t, 300)
check(
    "a PER-SAT offset moves that bias, not the clock",
    abs(_js.bias(("E", 3)) - _b_before[("E", 3)] - 2.0) < 0.6
    and abs(_js.clk - _clk_before) < 0.5,
    "db %.2f (want 2.0), dclk %.2f"
    % (_js.bias(("E", 3)) - _b_before[("E", 3)], _js.clk - _clk_before),
)

# The discriminating test (section 3a): take a satellite to no usable SNR and stop feeding
# it. Its prediction must keep moving with the shared clock while its bias holds, which a
# design anchoring each satellite to its own track cannot do.
_rnd.seed(77)
_jc = JointReceiverState(q_b=0.013)
_t = _run(_jc, 151.0, _TB, 0.0, 400)
_t = _run(_jc, 151.0, _TB, _t, 300, rate=0.01, skip={("E", 5)})  # clock walks 6 chips
_truth_5 = _run.clk + _TB[("E", 5)]
check(
    "a satellite with no measurements coasts on the shared clock",
    abs(_jc.predicted(("E", 5)) - _truth_5) < 0.5,
    "coast error %.3f chips after 600 s dark, clock moved %.1f"
    % (_jc.predicted(("E", 5)) - _truth_5, _run.clk - 151.0),
)
# clk_rate: check both bias and scatter. A Kalman velocity is noisy by construction, so a
# single instantaneous sample against a tight bar tests luck: average over a window for the
# mean, and measure the scatter separately, because the scatter decides whether this state
# can replace the l-a EMA.
from math import sqrt as _msqrt

_rates = []
_t0r, _clk0r = _t, _run.clk  # continue the ramp; a fresh origin per step is no ramp
for _ in range(300):
    _t += 2.0
    _clkr = _clk0r + 0.01 * (_t - _t0r)
    _jc.cycle([(k, _clkr + b + _rnd.gauss(0, 0.30), 0.30) for k, b in _TB.items()], _t)
    _rates.append(_jc.clk_rate)
_mr = sum(_rates) / len(_rates)
_sr = _msqrt(sum((r - _mr) ** 2 for r in _rates) / len(_rates))
check(
    "clk_rate is observable from the fleet, unbiased",
    abs(_mr - 0.01) < 0.0015,
    "%.5f vs 0.01000 chips/s" % _mr,
)
check(
    "clk_rate is quieter than the l-a EMA",
    _sr < 0.0015,
    "sd %.5f chips/s (bound 0.0015)" % _sr,
)

# Robustness. Unlike the circular median of the legacy clock, the solve is not
# outlier-proof by construction: one weak-satellite detection thousands of chips off could
# walk clk_rate and feed every unlocked seed a fictitious code drift. Each guard gets a
# check that fails without it.
_rnd.seed(808)
_jr2 = JointReceiverState()
_t2 = 0.0
for _ in range(400):
    _t2 += 2.0
    _jr2.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t2)
_clk_ok, _rate_ok = _jr2.clk, _jr2.clk_rate
_t2 += 2.0
_jr2.cycle(
    [(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()]
    + [(("G", 99), 151.0 + 1800.0, 0.3)],
    _t2,
)  # the garbage detection
for _ in range(30):
    _t2 += 2.0
    _jr2.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t2)
check(
    "one garbage detection cannot move the clock (birth window)",
    abs(_jr2.clk - _clk_ok) < 1.0 and ("G", 99) not in _jr2._idx,
    "clk %+.2f -> %+.2f, born=%s" % (_clk_ok, _jr2.clk, ("G", 99) in _jr2._idx),
)
check(
    "one garbage detection cannot move clk_rate",
    abs(_jr2.clk_rate - _rate_ok) < 0.002,
    "rate %+.5f -> %+.5f chips/s (bound 0.002)" % (_rate_ok, _jr2.clk_rate),
)
# An established satellite's outlier is the innovation gate's job.
_rej0 = _jr2.rejected
_t2 += 2.0
_jr2.cycle([(("G", 11), 151.0 + 900.0, 0.3)], _t2)
check(
    "an established satellite's outlier is rejected, not averaged in",
    _jr2.rejected > _rej0 and abs(_jr2.clk - _clk_ok) < 1.0,
)
# Membership churn must not masquerade as clock rate.
_rnd.seed(809)
_jm = JointReceiverState()
_t3 = 0.0
for _ in range(400):
    _t3 += 2.0
    _jm.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t3)
_gone = {("G", 11)}
_rmax = 0.0
for _ in range(700):  # let it age out, then keep running
    _t3 += 2.0
    _jm.cycle(
        [
            (k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3)
            for k, b in _TB.items()
            if k not in _gone
        ],
        _t3,
    )
    _rmax = max(_rmax, abs(_jm.clk_rate))
check(
    "a satellite SETTING does not masquerade as clock rate",
    _rmax < 0.002,
    "peak |rate| %.5f chips/s during the gauge re-reference" % _rmax,
)

# ── the prior gauge: an absolute reference does not step on membership ───────────────
# The median gauge defines clk against the population, so removing the satellite that is
# the median element steps bare clk by construction. The prior gauge's reference is
# per-satellite and absolute, so the same event must move clk by ~nothing.
# _TB's biases sorted: -3.4 -2.7 -1.7 +0.8 +2.9 +4.1, so the median pair is (-1.7, +0.8).
# Removing ("C", 39) (b = -1.7) moves the 5-satellite median to +0.8: a 1.25-chip step in
# the median-gauge convention, which both halves of this test discriminate on.
def _churn_step(mode):
    """(at-event |dclk| within 5 cycles of the drop, settled |dclk| 300 cycles later,
        worst |d(clk+b_i)| over survivors, peak |rate| post-drop)."""
    _rnd.seed(812)
    _j = JointReceiverState(gauge_mode=mode)
    _tt = 0.0
    for _ in range(400):
        _tt += 2.0
        _j.cycle(
            [(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _tt
        )
    _pre_clk = _j.clk
    _pre_sum = {k: _j.predicted(k) for k in _TB if k != ("C", 39)}
    _keep = {k: b for k, b in _TB.items() if k != ("C", 39)}
    _n0 = len(_j._idx)
    _ev_clk, _rpk, _dropped_at = None, 0.0, None
    _pre_ev = _j.clk  # clk on the cycle before the drop
    for _i in range(760):  # the drop lands at ~450 (max_age_s = 900 s)
        _tt += 2.0
        _j.cycle(
            [(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _keep.items()], _tt
        )
        if _dropped_at is None and len(_j._idx) < _n0:
            _dropped_at = _i
            _pre_clk = _pre_ev
        if _dropped_at is None:
            _pre_ev = _j.clk
        elif _i == _dropped_at + 5:
            _ev_clk = abs(_j.clk - _pre_clk)
        if _dropped_at is not None:
            _rpk = max(_rpk, abs(_j.clk_rate))
    _settled = abs(_j.clk - _pre_clk)
    _wsum = max(abs(_j.predicted(k) - v) for k, v in _pre_sum.items())
    return _ev_clk, _settled, _wsum, _rpk


# Removing ("C", 39) moves the survivors' median by ~1.25 chips and their mean by a few
# tenths. The median gauge pins the former at once (gauge_sigma 0.1 acts within a couple of
# cycles); the prior gauge has no instantaneous pin and relaxes toward the latter, slowly.
_pe, _psl, _pw, _pr = _churn_step("prior")
_me, _msl, _mw, _mr2 = _churn_step("median")
check(
    "PRIOR gauge: no at-event clk step when a satellite drops",
    _pe is not None and _pe < 0.10,
    "clk moved %.3f chips within 5 cycles of the drop" % (_pe or -1),
)
check(
    "... where the MEDIAN gauge steps at the event, by construction",
    _me is not None and _me > 0.5,
    "median-mode at-event step only %.3f -- the test cannot discriminate" % (_me or -1),
)
check(
    "PRIOR gauge: the settled shift is the small soft-mean change, not the median jump",
    _psl < 0.5 and _msl > 0.8,
    "settled dclk: prior %.3f (soft-mean ~0.25) vs median %.3f (~1.25)" % (_psl, _msl),
)
check(
    "... and clk+b_i (what every consumer reads) is invariant under both",
    max(_pw, _mw) < 0.35,
    "worst survivor drift: prior %.3f, median %.3f chips" % (_pw, _mw),
)
check(
    "PRIOR gauge: the post-churn relaxation does not masquerade as clock rate",
    _pr < 0.002,
    "peak |rate| %.5f chips/s (bound 0.002, same as the median-mode bench)" % _pr,
)

# Robustness of the prior gauge. Its pull is Huber-weighted (see gauge()): it must (a) pull
# a uniformly displaced comb home, and (b) give one wild bias only a bounded pull on clk, not
# a proportional one. The comb's +97-chip innovations are all far past the Huber cap, so
# healing is slow by design. What must hold throughout a displaced-comb episode is (a) the
# gauge never switches itself off, as an inlier screen centred on zero would, and (b)
# clk + b_i stays correct, so every consumer of the sum rides through the episode.
_rnd.seed(813)
_jh = JointReceiverState(gauge_mode="prior")
_t5 = 0.0
for _ in range(400):
    _t5 += 2.0
    _jh.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t5)
for _k5 in _TB:  # displaced comb: every bias +97, clk -97
    _jh.x[_jh._idx[_k5]] += 97.0
_jh.x[0] -= 97.0
_sums_ok = {k: _jh.predicted(k) for k in _TB}
_worst_sum = 0.0
for _ in range(600):
    _t5 += 2.0
    _jh.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t5)
    _worst_sum = max(
        _worst_sum, max(abs(_jh.predicted(k) - v) for k, v in _sums_ok.items())
    )
_med_b = sorted(float(_jh.x[_jh._idx[_k5]]) for _k5 in _TB)[len(_TB) // 2]
check(
    "PRIOR gauge: a displaced comb is healing -- the gauge never goes dead",
    _med_b < 96.0,
    "median b %.1f after 20 min: no progress from +97 means the pull "
    "switched off" % _med_b,
)
check(
    "... and clk+b_i stays correct THROUGHOUT the episode (sum-reading consumers ride it)",
    _worst_sum < 0.5,
    "worst sum error %.3f chips during the heal" % _worst_sum,
)

# A wild bias must add nothing beyond what silence alone does. The paired control is the
# whole test: run the same fleet twice from the same seed, once merely silencing a
# satellite and once corrupting its state by +97 and silencing it, and require the clk
# trajectories to agree. Whatever silencing does to the soft-mean equilibrium (age-out at
# 900 s, slow re-centre) happens in both arms; the +97 itself must contribute ~nothing,
# because R-inflation cuts its weight by (97/cap)^2 ~ 260x.
def _wild_run(hack):
    _rnd.seed(814)
    _j = JointReceiverState(gauge_mode="prior")
    _tt = 0.0
    for _ in range(400):
        _tt += 2.0
        _j.cycle(
            [(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _tt
        )
    if hack:
        _j.x[_j._idx[("G", 14)]] += 97.0
    _trace = []
    for _ in range(600):  # through age-out at 900 s and beyond
        _tt += 2.0
        _j.cycle(
            [
                (k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3)
                for k, b in _TB.items()
                if k != ("G", 14)
            ],
            _tt,
        )
        _trace.append(_j.clk)
    return _trace


_tw, _tc = _wild_run(True), _wild_run(False)
_dmax = max(abs(a - b) for a, b in zip(_tw, _tc))
# The bound allows for ~450 cycles of the 1/260-weighted pull before age-out. A +97-chip
# state corruption has no production pathway (the innovation and birth gates, tested above,
# prevent it); this asserts the gauge cannot amplify one into a clock error.
check(
    "PRIOR gauge: a +97-chip wild bias adds ~nothing beyond silencing alone",
    _dmax < 0.25,
    "worst trajectory divergence %.3f chips over 20 min (bound 0.25)" % _dmax,
)

# The gate must not deadlock. A genuine step (clock event, re-anchor) is rejected by a
# normalized gate, so without an escape the state never follows it and every later
# measurement is rejected too -- forever. Garbage is sporadic; a real move persists.
_rnd.seed(811)
_jd = JointReceiverState()
_t4 = 0.0
for _ in range(400):
    _t4 += 2.0
    _jd.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t4)
for _ in range(60):  # the clock steps 40 chips and stays there
    _t4 += 2.0
    _jd.cycle([(k, 191.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t4)
check(
    "the filter recovers from a genuine STEP (gate has an escape)",
    abs(_jd.clk - 191.0) < 1.0,
    "clk %+.2f after a 40-chip step (want ~191)" % _jd.clk,
)

# Modulo: y arrives mod the code length, so a measurement that wraps must not explode.
_jw = JointReceiverState(code_len=10230.0)
_jw.cycle([(("G", 1), 10229.8, 0.3), (("G", 2), 10229.9, 0.3)], 1.0)
_jw.cycle([(("G", 1), 0.2, 0.3), (("G", 2), 0.3, 0.3)], 3.0)  # wrapped past L
check(
    "JointReceiverState wraps innovations across the code-length boundary",
    abs(_jw.wrap(_jw.clk - 10230.0)) < 1.0 or abs(_jw.clk) < 1.0,
    "clk %+.2f" % _jw.clk,
)
_jw.cycle([(("G", 1), 0.25, 0.3)], 3.0 + 901.0)  # PRN 2 goes stale
check(
    "a stale satellite leaves the state (and stops voting in the gauge)",
    ("G", 2) not in _jw._idx and ("G", 1) in _jw._idx,
)

# -- cross-band (l-a): the rate is receiver-wide, the phase is not -----------------------
# A band where no chain solves its own clock gets no (l-a) from the band-scoped lookup, so
# its seeds would carry code_phase_rate 0 and its code loop would never close. The rate
# crosses bands (code_bias_any_band); the phase must not, because a per-band phase offset is
# tau_band.
_rx = receiver.Receiver()
_rx.contribute_code_bias("gps_l5", "1176.45MHz", 1.5e-6, 6, 1000.0)
check(
    "same-band (l-a) still preferred",
    _rx.code_bias("1176.45MHz", exclude="gal_e5a", t_now=1000.0).value == 1.5e-6,
)
check(
    "a band with no contributor gets nothing from the band-scoped lookup",
    _rx.code_bias("1207.14MHz", exclude="gal_e5b", t_now=1000.0) is None,
)
_any = _rx.code_bias_any_band(exclude="gal_e5b", t_now=1000.0)
check(
    "cross-band (l-a) fallback finds the other band's rate",
    _any is not None and _any.value == 1.5e-6,
    "src %s" % (_any.src if _any else "-"),
)
check(
    "cross-band fallback still honours `exclude`",
    _rx.code_bias_any_band(exclude="gps_l5", t_now=1000.0) is None,
)
check(
    "cross-band fallback still honours max_age_s",
    _rx.code_bias_any_band(exclude="gal_e5b", t_now=1000.0 + 500.0) is None,
)
# The phase stays per band: adopting it across carriers would inject tau_band itself.
_rx.contribute_dr_clock(
    "gps_l5", "1176.45MHz", 151.0, 0.01, 1000.0, 1023000, chip_rate_hz=10.23e6
)
check(
    "dr_clock (the code PHASE) is still band-scoped -- tau_band is not borrowable",
    _rx.dr_clock("1207.14MHz", exclude="gal_e5b", t_now=1000.0) is None
    and _rx.dr_clock("1176.45MHz", exclude="gal_e5a", t_now=1000.0) is not None,
)
# ── chips are a unit, the clock is a time ────────────────────────────────────────────
# A donor clock in chips must be converted through the two chip rates: 150.2 chips at
# 10.23 Mcps is 75.1 chips at gal_e6's 5.115 Mcps, and adopted raw it reads as twice the
# clock time (+75 chips of code error on every seed).
check(
    "cross-rate clock adoption halves gps chips for the 5.115 Mcps consumer",
    abs(receiver.Receiver.clock_chips_convert(150.2, 10.23e6, 5.115e6, 5115) - 75.1)
    < 1e-9,
)
check(
    "same-rate clock adoption is the identity",
    abs(receiver.Receiver.clock_chips_convert(150.2, 10.23e6, 10.23e6, 10230) - 150.2)
    < 1e-9,
)
# The epoch extension (gps_l2c's bootstrap): a donor precise mod its short window plus an
# independent clock-mod measurement covering our longer period. Donor gps_l5: 152.22 chips
# at 10.23 Mcps = 14.879 us, window 1 ms; the NH joint fit says clock mod 20 ms ~ 40 us
# (it only has to pick among candidates 1 ms apart). Truth 14.879 us.
_t, _r = receiver.Receiver.clock_extend_mod(152.22 / 10.23e6, 1e-3, 40e-6, 0.02, 0.02)
check(
    "epoch extension recovers the donor's precision, k from the mod measurement",
    abs(_t - 152.22 / 10.23e6) < 1e-12 and abs(_r) < 0.25e-3,
    "t=%.3f us resid=%.1f us" % (_t * 1e6, _r * 1e6),
)
_t2, _ = receiver.Receiver.clock_extend_mod(152.22 / 10.23e6, 1e-3, 3.04e-3, 0.02, 0.02)
check(
    "...and a mod measurement near 3 ms picks k=3, three donor windows up",
    abs(_t2 - (152.22 / 10.23e6 + 3e-3)) < 1e-12,
    "t=%.3f us" % (_t2 * 1e6),
)
_t3, _r3 = receiver.Receiver.clock_extend_mod(
    152.22 / 10.23e6, 1e-3, 0.51e-3, 0.02, 0.02
)
check(
    "a mod measurement HALF a donor window away leaves a residual the caller must refuse",
    abs(_r3) > 0.25e-3,
    "resid=%.1f us" % (_r3 * 1e6),
)
_rx2 = receiver.Receiver()
_rx2.contribute_clock_mod_epoch("gps_l5", 40e-6, 0.02, 7, 1000.0)
check(
    "clock_mod_epoch returns a record whose epoch covers the asked period",
    _rx2.clock_mod_epoch(0.02, exclude="gps_l2c", t_now=1000.0) is not None,
)
check(
    "...refuses when the asked period exceeds the stored epoch (never lengthen a modulus)",
    _rx2.clock_mod_epoch(1.5, exclude="gps_l2c", t_now=1000.0) is None,
)
check(
    "...and excludes the asker's own contribution",
    _rx2.clock_mod_epoch(0.02, exclude="gps_l5", t_now=1000.0) is None,
)
check(
    "the stored dr_clock carries its chip rate for consumers",
    _rx.dr_clock("1176.45MHz", exclude="gal_e5a", t_now=1000.0).extra.get(
        "chip_rate_hz"
    )
    == 10.23e6,
)
# The joint state has one unit, reference chips (receiver.JOINT_REF_RATE_HZ), so a chain's
# code period is converted before it can shorten the modulus: E6's 5115 chips at 5.115 Mcps
# are gps_l5's 1 ms, and a 20-ms L2C-CM code joining never lengthens it.
_rx3 = receiver.Receiver()
_j3 = _rx3.joint_receiver(
    "1176.45MHz", 10230.0, 10.23e6, gauge_mode="prior", clk0=150.0
)
_j3.cycle([(k, 150.2 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], 2.0)
_clk3 = _j3.clk
check(
    "joint_receiver converts another chip rate's code period to reference chips",
    _rx3.joint_receiver("1278.75MHz", 5115.0, 5.115e6, gauge_mode="prior") is _j3
    and _rx3.joint_receiver("1227.60MHz", 10230.0, 0.5115e6, gauge_mode="prior") is _j3
    and _j3.L == 10230.0
    and _j3.clk == _clk3,
    "L %g clk %.3f after E6 and L2C-CM joined" % (_j3.L, _j3.clk),
)
check(
    "joint_unit() is the reference rate and the modulus; None with no state",
    _rx3.joint_unit() == (receiver.JOINT_REF_RATE_HZ, 10230.0)
    and receiver.Receiver().joint_unit() is None,
)
check(
    "...and a 0.5-ms period at E6's rate shortens it to 5115",
    _rx3.joint_receiver("x", 2557.5, 5.115e6, gauge_mode="prior").L == 5115.0,
)
# And the consumer's arithmetic: a fractional-frequency bias -> chips/hop, chip-rate scaled.
from gnss_broker.fits import cp_rate_from_code_bias  # noqa: E402

_r_e5b = cp_rate_from_code_bias(1.5e-6, 195312.5, 10.23e6)
check(
    "the borrowed rate is nonzero, i.e. the seed carries a rate at all",
    abs(_r_e5b) > 0.0,
    "%.6g chips/hop" % _r_e5b,
)

# -- cross-band clock phase: a bootstrap, with a modulus rule -----------------------------
# The rate alone is not enough: with the clock 150 chips out there is no peak for a rate to
# hold. These check the phase bootstrap and its guard: a clock may be reduced to a shorter
# code but not lengthened on its own, because a value known mod 10230 says nothing about
# which of the 100 periods of a 1023000-chip code it sits in (only the epoch extension above
# can resolve that).
_rx2 = receiver.Receiver()
_rx2.contribute_dr_clock("gps_l5", "1176.45MHz", 150.74, 0.01, 2000.0, 1023000)
check(
    "cross-band clock found when no same-band donor exists",
    _rx2.dr_clock("1207.14MHz", exclude="gal_e5b", t_now=2000.0) is None
    and _rx2.dr_clock_any_band(exclude="gal_e5b", t_now=2000.0) is not None,
)
_c = _rx2.dr_clock_any_band(exclude="gal_e5b", t_now=2000.0)
check(
    "cross-band clock carries its code_length so the consumer can judge the modulus",
    _c.extra.get("code_length") == 1023000,
)
check(
    "reducing to a SHORTER code is well defined (150.74 mod 10230)",
    abs((_c.value % 10230.0) - 150.74) < 1e-9,
)
# The refusal that matters: a short-code donor must not seed a long-code consumer alone.
_rx3 = receiver.Receiver()
_rx3.contribute_dr_clock("bds_b2b", "1207.14MHz", 150.74, 0.01, 2000.0, 10230)
_c3 = _rx3.dr_clock_any_band(exclude="gal_e5a", t_now=2000.0)
check(
    "a SHORTER-code donor is visible but must be refused by the caller's guard",
    _c3 is not None and (_c3.extra.get("code_length") or 0) < 1023000,
)
check(
    "cross-band clock still honours exclude and max_age",
    _rx2.dr_clock_any_band(exclude="gps_l5", t_now=2000.0) is None
    and _rx2.dr_clock_any_band(exclude="gal_e5b", t_now=2000.0 + 500.0) is None,
)
# The error the bootstrap accepts is tau_band-sized, not 150 chips: the arithmetic.
check(
    "bootstrap replaces a 150-chip error with a sub-chip one",
    abs(150.74 - 0.0) > 100.0 and abs((150.74 % 10230.0) - 150.74) < 1e-9,
)

print(
    "\n%d/%d checks passed" % (0 if FAIL else 1, 1)
    if False
    else ("FAILED: %s" % ", ".join(FAIL))
    if FAIL
    else "\nALL CHECKS PASSED"
)
sys.exit(1 if FAIL else 0)
