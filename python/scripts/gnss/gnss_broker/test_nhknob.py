"""The per-PRN overlay-period knob: a whole-period shift on the wire, nothing else.
    python3 -m gnss_broker.test_nhknob
Pinned: only the named PRNs move; both the argument and the physical phase move by exactly
k*code_len modulo the tiled length; a phase of -1 (absent) stays -1; k = 0 or an empty table
is a no-op that returns the caller's own list; negative k wraps.
"""
import sys
from gnss_broker.seeding import apply_nh_prn_offset
_fails = []
def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok: _fails.append(what)
CL, SEG = 10230.0, 20
pay = [dict(prn=8, code_phase_chips=100.5, code_phase_at_ref_chips=200.25, doppler_hz=1.0),
       dict(prn=10, code_phase_chips=300.0, code_phase_at_ref_chips=-1.0, doppler_hz=2.0),
       dict(prn=18, code_phase_chips=204000.0, code_phase_at_ref_chips=203990.0)]
same = apply_nh_prn_offset(pay, {}, CL, SEG)
check(same is pay, "empty table returns the caller's list untouched")
check(apply_nh_prn_offset(pay, {8: 0}, CL, SEG)[0] is pay[0], "k = 0 leaves the seed object alone")
out = apply_nh_prn_offset(pay, {8: 3, 18: 1}, CL, SEG)
check(out[0]["code_phase_chips"] == 100.5 + 3 * CL and out[0]["code_phase_at_ref_chips"] == 200.25 + 3 * CL,
      "PRN 8: argument and phase both move by 3 periods")
check(out[0]["doppler_hz"] == 1.0 and pay[0]["code_phase_chips"] == 100.5, "other fields and the input untouched")
check(out[1] is pay[1], "PRN 10 not in the table: same object")
check(abs(out[2]["code_phase_chips"] - ((204000.0 + CL) % (CL * SEG))) < 1e-9
      and abs(out[2]["code_phase_at_ref_chips"] - ((203990.0 + CL) % (CL * SEG))) < 1e-9, "PRN 18 wraps at the tiled length")
neg = apply_nh_prn_offset(pay, {10: -2}, CL, SEG)
check(abs(neg[1]["code_phase_chips"] - ((300.0 - 2 * CL) % (CL * SEG))) < 1e-9 and neg[1]["code_phase_at_ref_chips"] == -1.0,
      "negative k wraps; an absent phase (-1) stays -1")
check(apply_nh_prn_offset(pay, {8: 5}, CL, 1) is pay, "a single-period code (lc_seg 1) has no overlay to shift")
print("%d check(s), %d failed" % (8, len(_fails)))
sys.exit(1 if _fails else 0)
