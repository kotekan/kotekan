"""Is a chain's search-anchored clock solve established?

The dead-reckon clock stage keeps, per detected satellite, the integrity residual of the
search's code phase against the model with the solved clock removed
(`dr_state["integ"][prn] = (chips, t)`). A solve is established when at least two fresh
residuals sit inside a chip and none sits far off: one satellite is unfalsifiable, two is the
minimum that can disagree. Once a chain has a clock and drift, one with nothing fresh to solve
(no detections, or a collapsed sky with no predictions) is not established, and neither is one
locked on noise, whose residuals scatter over the code. A chain that has never solved makes no
claim.

While a chain's solve is not established, no other chain takes its clock (deadreckon's
cross-band bootstrap and same-band adoption), its NH search hint learns no overlay offset
(searchhint), and its clock mod the overlay epoch is not published (seeding). A chain locked
on noise would otherwise hand another chain a clock whole code periods off, and the hint would
learn the same wrong epoch.
"""


def dr_solve_healthy(
    dr_state, now, max_age_s=60.0, max_chips=1.0, min_sats=2, max_any_chips=5.0
):
    """True = established, False = not (stale, thin, or a residual far off), None = never
    judged: a chain that has not solved at all yet makes no claim, so a fresh broker's first
    bootstrap is not held up by it. Consumers refuse on False only.

    Established = at least `min_sats` residuals younger than `max_age_s` inside `max_chips`,
    and none beyond `max_any_chips` (default 5 chips, the clock-adoption bound the chains yaml
    sets with --dr-clock-adopt-max-chips). The wide outer bound lets an established solve carry
    a satellite or two at 1-2 chips; on noise the residuals scatter over the whole code."""
    if not dr_state or "integ" not in dr_state:
        return None
    integ = dr_state.get("integ") or {}
    resid = [abs(v[0]) for v in integ.values() if now - v[1] <= max_age_s]
    inside = sum(1 for r in resid if r <= max_chips)
    return inside >= min_sats and max(resid, default=0.0) <= max_any_chips
