"""Is a chain's search-anchored clock solve established? (#161)

The dead-reckon clock stage keeps, per detected satellite, the integrity residual of the
search's code phase against the solved clock (`dr_state["integ"][prn] = (chips, t)`). A solve
is established when at least two fresh residuals sit inside a chip and none outside it (one
satellite is unfalsifiable; the #152 latch detector asks for three, this gate asks for the
minimum that can be wrong). A chain with nothing to solve (no detections, or a
collapsed sky with no predictions) is not established, and neither is one locked on noise,
whose residuals scatter over the code. The 2026-10-08 incident was both in turn: gps_l5
bootstrapped gps_l2c's clock 10 ms off while on noise, and the hint learned the same wrong
epoch from the noise detections.
"""


def dr_solve_healthy(dr_state, now, max_age_s=60.0, max_chips=1.0, min_sats=2):
    """True = established, False = not (stale, thin, or a residual outside a chip), None =
    never judged: a chain that has not solved at all yet makes no claim, so a fresh broker's
    first bootstrap is not held up by it. Consumers refuse on False only."""
    if not dr_state or "integ" not in dr_state:
        return None
    integ = dr_state.get("integ") or {}
    resid = [abs(v[0]) for v in integ.values() if now - v[1] <= max_age_s]
    return len(resid) >= min_sats and max(resid) <= max_chips
