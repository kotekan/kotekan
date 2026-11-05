"""gw_nav_frame: the GuoWang frame index the seed ships agrees with the reduced code phase the
same model ships, follows the at-ref phase across a wrap, and is exact on the large GPST."""
import math
from fractions import Fraction

from gnss_broker.deadreckon import (
    GW_NAV_FRAME_CHIPS,
    GW_NAV_FRAME_S,
    GW_NAV_WEEK_S,
    gw_nav_frame,
    t_mod_code,
)

CHIP = 2046000.0
GPST0 = 1478e6 + 0.123456789  # a week in, sample-0 GPST of a CHORD uptime


def _model(t_ref, range_m, clk, off):
    """cp_predicted's remainder: (t_tx mod 2 ms) x chip rate, built as the broker builds it."""
    t0m = t_mod_code(GPST0, 1, GW_NAV_FRAME_CHIPS, CHIP)
    t_tx = t0m + t_ref - range_m / 299792458.0 + clk
    return ((t_tx % GW_NAV_FRAME_S) * CHIP + off) % GW_NAV_FRAME_CHIPS


def test_frame_and_phase_describe_one_instant():
    for t_ref, rng, clk, off in (
        (12.5, 1.2e6, -8.6e-6, 3.0),
        (3600.25, 2.4e6, 3.1e-4, -11.5),
        (86399.9, 9.5e5, 0.0, 0.0),
        (604799.999, 1.8e6, -5e-6, 4091.9),
    ):
        frame, phase = gw_nav_frame(GPST0, t_ref, rng, clk, off, CHIP)
        # the phase within the frame is the broker's reduced phase (to a chip rounding)
        assert abs(((phase - _model(t_ref, rng, clk, off) + 2046) % 4092) - 2046) < 0.01
        # the frame is the exact floor of the BDT-week transmit time
        t_tx = (
            Fraction(GPST0)
            + Fraction(t_ref)
            - Fraction(rng) / 299792458
            + Fraction(clk)
            + Fraction(off) / Fraction(int(CHIP))
            - 14
        ) % GW_NAV_WEEK_S
        assert frame == int((t_tx / Fraction(GW_NAV_FRAME_S)) // 1)
        assert 0 <= frame < GW_NAV_WEEK_S / GW_NAV_FRAME_S


def test_shipped_phase_moves_the_frame_across_a_wrap():
    frame, phase = gw_nav_frame(GPST0, 100.0, 1.0e6, 0.0, 0.0, CHIP)
    # a shipped phase a few chips below the model's, just before the model's frame start
    below = (phase - 3.0) % GW_NAV_FRAME_CHIPS
    f2, p2 = gw_nav_frame(
        GPST0, 100.0, 1.0e6, 0.0, 0.0, CHIP, shipped_phase_chips=below
    )
    assert abs(p2 - below) < 1e-9
    assert f2 == frame - (1 if phase < 3.0 else 0)
    above = (phase + 3.0) % GW_NAV_FRAME_CHIPS
    f3, p3 = gw_nav_frame(
        GPST0, 100.0, 1.0e6, 0.0, 0.0, CHIP, shipped_phase_chips=above
    )
    assert f3 == frame + (1 if phase + 3.0 >= GW_NAV_FRAME_CHIPS else 0)


def test_frames_advance_at_one_per_two_ms():
    f0, p0 = gw_nav_frame(GPST0, 50.0, 1.0e6, 0.0, 0.0, CHIP)
    f1, p1 = gw_nav_frame(GPST0, 50.0 + 1.0, 1.0e6, 0.0, 0.0, CHIP)
    assert f1 - f0 == 500
    assert abs(p1 - p0) < 1e-6
