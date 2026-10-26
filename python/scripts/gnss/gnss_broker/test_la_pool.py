"""The code-rate clock (l-a): what may feed it, and when it holds.

    python3 -m gnss_broker.test_la_pool     (from python/scripts/gnss)

10-26 05:38Z: G13 crossed 0.12 deg from boresight, the search reported noise code phases for
the other satellites, their cp fits were flushed and rebuilt from 3 points, and the pool took a
2-sample median of +1.44 ppm (one 3-point fit, one honest). The EMA reached +0.12 ppm, ~1.2
chips/s on every chain's seeds. These checks pin the four rules that stop that:
  * the sample gate (seeding): only mature fits (>= 6 points over fit_maturity_span_s), and a
    rate the cp-rate tolerance refused goes to la_rejected, never to the pool;
  * min sats: two fits are not a median;
  * the transit hold: no update (and no bootstrap) inside the freeze cone;
  * the re-pin: an EMA that is itself wrong cannot refuse honest fits for ever.

@author Keith Vanderlinde
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker.clockbias import ClockBias  # noqa: E402
from gnss_broker.test_escape_dedup import HPS, L, STEP, seeding_ctx, run  # noqa: E402

_fails = []
PPM = 1e-6
CHIP = 10.23e6
TOL = 0.5 / CHIP  # --cp-rate-model-tol 0.5 chips/s as a fraction
KW = dict(min_sats=3, max_frac=3.0 * PPM, alpha=0.05, tol_frac=TOL, repin_cycles=10)


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def gate(rate_chips_s, n, code_ema=None, tol=0.0):
    """Drive stage_detections_to_seeds with n detections 10 s apart on a code phase moving at
    rate_chips_s; returns [(len(la_samples), len(la_rejected))] per cycle."""
    ctx = seeding_ctx(cp_rate_model_tol=tol)
    ctx.cb.code_ema = code_ema
    out = []
    h0 = 20_000_000
    dets = [h0 + i * STEP for i in range(n)]
    cp_at = lambda h: (4321.0 + rate_chips_s * (h - h0) / HPS) % L  # noqa: E731
    for i in range(n):
        run(ctx, 7, dets[: i + 1][-1:], cp_at, t0=1000.0 + 10.0 * i)
        out.append((len(ctx.la_samples), len(ctx.la_rejected)))
    return out


def main():
    print("sample gate (seeding.stage_detections_to_seeds)")
    g = gate(0.0, 8)
    check(
        all(a == 0 for a, _ in g[:5]),
        "fits of 3..5 points (or under 30 s) feed nothing: %s" % g[:5],
    )
    check(
        g[5][0] == 1 and g[7][0] == 1,
        "a mature fit (6 points over 50 s) feeds one sample: %s" % g,
    )
    g = gate(2.0, 8, code_ema=0.0, tol=0.5)
    check(
        all(a == 0 for a, _ in g) and g[7][1] == 1,
        "a mature fit 2 chips/s off the clock (tol 0.5) goes to la_rejected, not the pool: %s"
        % g,
    )

    print("pool (ClockBias.update_code_rate)")
    cb = ClockBias()
    act, raw = cb.update_code_rate([-0.001 * PPM, 2.875 * PPM], [], None, **KW)
    check(act == "few" and cb.code_ema is None, "05:38Z's two fits are not a median")
    act, raw = cb.update_code_rate(
        [-0.002 * PPM, -0.001 * PPM, -0.003 * PPM], [], None, **KW
    )
    check(
        act == "update" and abs(cb.code_ema + 0.002 * PPM) < 1e-15,
        "three fits bootstrap it",
    )
    cb.update_code_rate([0.0, 0.0, 0.0], [], None, **KW)
    check(abs(cb.code_ema + 0.0019 * PPM) < 1e-15, "then an EMA at alpha 0.05")
    e = cb.code_ema
    act, _ = cb.update_code_rate([1.4 * PPM] * 4, [], (0.12, ("G", 13)), **KW)
    check(
        act == "hold" and cb.code_ema == e,
        "inside a transit: held, whatever the fits say",
    )
    check(cb.code_hold == (0.12, ("G", 13)), "and it says which satellite")
    act, _ = cb.update_code_rate([0.0] * 3, [], None, **KW)
    check(
        act == "update" and cb.code_hold is None,
        "the transit clears and it updates again",
    )
    act, raw = cb.update_code_rate([5.0 * PPM] * 3, [], None, **KW)
    check(
        act == "refused" and abs(cb.code_ema) < 0.01 * PPM,
        "a median beyond 3 ppm is refused",
    )
    cb2 = ClockBias()
    act, _ = cb2.update_code_rate([0.1 * PPM] * 3, [], (1.0, ("E", 16)), **KW)
    check(act == "hold" and cb2.code_ema is None, "no bootstrap inside a transit")

    print("re-pin")
    cb = ClockBias()
    cb.code_ema = (
        0.2 * PPM
    )  # wrong: the true rate is ~0, and every honest fit is now refused
    acts = [
        cb.update_code_rate([], [0.001 * PPM, -0.001 * PPM, 0.0], None, **KW)[0]
        for _ in range(10)
    ]
    check(
        acts[:9] == ["few"] * 9 and acts[9] == "repin",
        "10 cycles of agreeing refusals re-pin",
    )
    check(abs(cb.code_ema) < 1e-15, "to their median")
    cb = ClockBias()
    cb.code_ema = 0.2 * PPM
    for i in range(12):
        rej = (
            [0.0, 0.001 * PPM, 0.3 * PPM]
            if i == 5
            else [0.0, 0.001 * PPM, -0.001 * PPM]
        )
        act, _ = cb.update_code_rate([], rej, None, **KW)
    check(
        act == "few" and cb.code_ema == 0.2 * PPM,
        "a cycle whose refusals scatter restarts the run",
    )
    cb = ClockBias()
    cb.code_ema = 0.2 * PPM
    for i in range(12):
        acc = [0.2 * PPM] * 3 if i == 5 else []
        act, _ = cb.update_code_rate(acc, [0.0, 0.0, 0.0], None, **KW)
    check(
        act == "few" and abs(cb.code_ema - 0.2 * PPM) < 1e-15,
        "an accepted cycle restarts it too",
    )
    cb = ClockBias()
    cb.code_ema = 0.2 * PPM
    for _ in range(12):
        act, _ = cb.update_code_rate([], [0.0, 0.0], None, **KW)
    check(cb.code_ema == 0.2 * PPM, "two refusals are not evidence")

    if _fails:
        raise SystemExit("FAILED: %d" % len(_fails))
    print("ok")


if __name__ == "__main__":
    main()
