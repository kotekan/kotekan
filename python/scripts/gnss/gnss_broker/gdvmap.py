"""The array's direction-dependent code group delay (GDV), as a delay term in the range model.

A satellite's code arrives later than its geometric range says by an amount that depends on the
direction it arrives from relative to the dishes: ~+8.5 m north low (behind the dish at the Tau A
tilt), ~+2 m east/west low, ~0 south and high, the same on every band (frequency-flat to ~1 m).
It is a property of the element pattern (feed and coupling to the neighbouring dishes, identical
for every dish since they tip together), measured like an IGS antenna PCV/GDV calibration and
applied the same way: as a one-way delay beside tropo_m and iono_m, not as a per-satellite
constant (memory chord-c41-north-low-code-bias-20261026; fit python/scripts/gnss/gdv_fit.py).

The map is fitted per band on a sky grid (az 5 x el 2.5 deg below el 20, az 10 x el 5 above)
jointly with one constant per satellite, relative to each epoch's median satellite, gauge
median M(el >= 45) = 0. A constant common to every satellite is the receiver clock's, so the
gauge costs nothing. Sign: + = LATE (a longer path), the sense of tropo_m and iono_m, so
cp_predicted adds gdv_m to the range. The obs rows' code_resid_m (cp_phys - cp_model, negative =
late) loses it once applied; each row carries gdv_m so the pre-map residual is code_resid_m -
gdv_m.

Lookup is a Gaussian-kernel mean over the fitted cells' centres on the sphere (KERNEL_DEG), so
the term is smooth as a satellite crosses cell edges and the two grid resolutions join without a
seam. A direction with no fitted cell within SUPPORT_DEG gets 0.0 (no correction), never an
extrapolation.

The map belongs to one pointing. It applies only while the installed boresight (sky.py, from the
nodes' telescope.dish_coelev_deg) is within BORESIGHT_TOL_DEG of the map's own; otherwise every
row gets 0.0 and the state is logged once per change.

    python3 -m gnss_broker.gdvmap export FIT.json OUT.json --pointing P --az 180 --el 62.7

writes the slim map file the broker reads from gdv_fit.py's output.
"""

import json
import math
import os

KERNEL_DEG = 2.0  # Gaussian sigma on the sky; the north-low gradient is ~1 m/deg in el
SUPPORT_DEG = 5.5  # nearest fitted cell further than this: no correction; > the half-diagonal of a 10 x 5 deg cell
BORESIGHT_TOL_DEG = 0.5
_REACH_DEG = 3.0 * KERNEL_DEG
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def cell_centre(key):
    """(az, el) centre of a fitted cell from its key "az_lo,el_lo" (gdv_fit.cell)."""
    a, e = key.split(",")
    az, el = float(a), float(e)
    if el < 20.0:
        return az + 2.5, el + 1.25
    return az + 5.0, el + 2.5


def _unit(az_deg, el_deg):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return (math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e))


def sep_deg(az1, el1, az2, el2):
    u, v = _unit(az1, el1), _unit(az2, el2)
    return math.degrees(
        math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(u, v)))))
    )


def resolve(path):
    """An absolute path as given; a relative one against the repository root, so the chains yaml
    names a tracked file the same way in either repository."""
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


class GdvMap:
    """One band's map: delay_m(az, el) in metres, + = late, or 0.0 off the fitted sky."""

    def __init__(self, doc, band):
        if band not in doc.get("bands", {}):
            raise ValueError(
                "gdv map %s has no band %r (has %s)"
                % (doc.get("pointing"), band, ", ".join(sorted(doc.get("bands", {}))))
            )
        b = doc["bands"][band]
        self.band = band
        self.pointing = doc["pointing"]
        self.bore_az = float(doc["boresight"]["az"])
        self.bore_el = float(doc["boresight"]["el"])
        self.cells = []
        for key, m in b["M"].items():
            az, el = cell_centre(key)
            self.cells.append((az, el, _unit(az, el), float(m)))
        if not self.cells:
            raise ValueError("gdv map %s band %s has no cells" % (self.pointing, band))

    @classmethod
    def load(cls, path, band):
        with open(resolve(path)) as f:
            return cls(json.load(f), band)

    def applies(self, bore_az, bore_el):
        """Is the installed boresight this map's pointing? False when none is installed."""
        if bore_az is None or bore_el is None:
            return False
        return (
            sep_deg(bore_az, bore_el, self.bore_az, self.bore_el) <= BORESIGHT_TOL_DEG
        )

    def delay_m(self, az, el):
        u = _unit(az, el)
        c_reach = math.cos(math.radians(_REACH_DEG))
        c_support = math.cos(math.radians(SUPPORT_DEG))
        w_sum = m_sum = 0.0
        near = False
        for _az, _el, v, m in self.cells:
            # cheap reject on elevation before the dot product
            if abs(_el - el) > _REACH_DEG:
                continue
            c = sum(x * y for x, y in zip(u, v))
            if c < c_reach:
                continue
            if c >= c_support:
                near = True
            d = math.degrees(math.acos(min(1.0, c)))
            w = math.exp(-0.5 * (d / KERNEL_DEG) ** 2)
            w_sum += w
            m_sum += w * m
        if not near or w_sum <= 0.0:
            return 0.0
        return m_sum / w_sum


def export(fit_path, out_path, pointing, bore_az, bore_el):
    """gdv_fit.py's output -> the slim map: M and sigma per band, no per-satellite constants
    (those are what the map exists NOT to be)."""
    with open(fit_path) as f:
        fit = json.load(f)
    doc = {
        "what": "array code group-delay map (m, + = late), per band, relative to the epoch median "
        "satellite; gauge median M(el >= 45) = 0; cell key = az_lo,el_lo (az 5 x el 2.5 deg below "
        "el 20, az 10 x el 5 above); sigma = robust scatter about M + B per cell",
        "pointing": pointing,
        "boresight": {"az": bore_az, "el": bore_el},
        "source": {
            "fit": os.path.basename(fit_path),
            "tool": "python/scripts/gnss/gdv_fit.py",
            "train": fit.get("train"),
            "test": fit.get("test"),
            "fit_pointing": fit.get("pointing"),
        },
        "bands": {
            k: {
                "M": {c: round(m, 3) for c, m in v["M"].items()},
                "sigma": {c: round(s, 3) for c, s in v.get("sigma", {}).items()},
                "split": v.get("split"),
            }
            for k, v in fit["bands"].items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    return doc


def _main(argv):
    import argparse

    ap = argparse.ArgumentParser(prog="gdvmap")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("fit")
    ex.add_argument("out")
    ex.add_argument("--pointing", required=True)
    ex.add_argument("--az", type=float, required=True)
    ex.add_argument("--el", type=float, required=True)
    q = sub.add_parser("query")
    q.add_argument("map")
    q.add_argument("band")
    q.add_argument("az", type=float)
    q.add_argument("el", type=float)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        doc = export(a.fit, a.out, a.pointing, a.az, a.el)
        print(
            "wrote %s: %s"
            % (
                a.out,
                ", ".join(
                    "%s %d cells" % (k, len(v["M"])) for k, v in doc["bands"].items()
                ),
            )
        )
    else:
        print("%.3f" % GdvMap.load(a.map, a.band).delay_m(a.az, a.el))


if __name__ == "__main__":
    import sys

    _main(sys.argv[1:])
