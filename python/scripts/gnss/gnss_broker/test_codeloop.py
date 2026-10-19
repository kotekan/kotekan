"""The DLL control loop holds the trim when a seed has no code_phase_rate, on both paths.

    python3 -m gnss_broker.test_codeloop

`stage_dll_control` integrates each seeded satellite's code trim from the fleet's
discriminator, or from one combiner's when the fleet has no row for the PRN. The trim absorbs
only what the seed's code_phase_rate feed-forward leaves, so with that rate absent or zero it
must hold (docs/CHORD_GNSS_SHARED_DLL.md section 7). This drives the stage on the
single-combiner path with the log captured and checks the hold, and that a live rate still
integrates.

@author Keith Vanderlinde
"""

import sys

from gnss_broker import codeloop


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class _Obj(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def run(seed):
    """One stage call on the single-combiner path (no fleet row) for PRN 5 with a 0.3-chip
    discriminator at SNR 100; returns (ctx, logged lines)."""
    lines = []
    ctx = _Obj(
        seeds={5: seed},
        status={5: {"dll_disc": 0.3}},
        dllp=_Obj(fleet={}, report=[]),
        dls=_Obj(trim={}, last={}, last_hop={}, armed_last=set()),
        sig_of=lambda rec: 100.0,
        args=_Obj(
            lock_snr=10.0,
            dll_gain=0.25,
            dll_leak=0.05,
            dll_leak_present=0.05,
            dll_spacing=0.5,
        ),
        dll_combiners=[],
    )
    saved = codeloop._log_rl
    codeloop._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
    try:
        codeloop.stage_dll_control(ctx)
    finally:
        codeloop._log_rl = saved
    return ctx, lines


def main():
    print("\nthe DLL rate gate")

    # 1. Single-combiner path, rate zero (a first search seed): hold, and say so.
    ctx, lines = run({"code_phase_rate": 0.0})
    check(5 not in ctx.dls.trim, "rate 0: no trim integrated (got %r)" % ctx.dls.trim)
    check(
        any("no live code_phase_rate" in m for m in lines), "rate 0: the hold is logged"
    )

    # 2. Single-combiner path, rate absent: hold.
    ctx, lines = run({})
    check(
        5 not in ctx.dls.trim, "rate absent: no trim integrated (got %r)" % ctx.dls.trim
    )

    # 3. A live rate integrates the discriminator.
    ctx, lines = run({"code_phase_rate": 3.45})
    check(
        ctx.dls.trim.get(5, 0.0) != 0.0,
        "live rate: the 0.3-chip disc integrates (trim %r)" % ctx.dls.trim.get(5),
    )

    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: the DLL rate gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
