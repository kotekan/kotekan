"""gnss_beam_cube.py's pixelisation, against healpy.

⚠️ WHY THIS TEST EXISTS AND WHY IT IS NOT OPTIONAL. gnss_beam_cube.py inlines ang2pix (RING)
and the RING/NEST pair so the builder runs in venv-ft rather than needing the healpy venv.
That is a reasonable trade only while the arithmetic is verified against the real thing: a
wrong pixelisation does not fail, it puts the beam somewhere else on the sky, and the result
still looks like a beam.

It has already earned its keep once. The first hand-written ring2nest was wrong for 100% of
pixels at every nside (nest2ring, written from the same reference, was exact) -- which made
the export's downgrade wrong for 80% of pixels and would have quietly scrambled every
multi-resolution cube. It is now the exact inverse permutation of the verified nest2ring,
which cannot be wrong in a way nest2ring is not.

Needs healpy: /home/kvand/gnss/venv/bin/python, NOT venv-ft. Skipped without it, and a skip
here is a real loss of coverage, not a pass.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "python", "scripts", "gnss"))

# RUNNABLE WITHOUT PYTEST, ON PURPOSE. No venv on this host currently has pytest, and healpy
# lives only in /home/kvand/gnss/venv -- so a pytest-only test here would be a test nobody
# can run, which is worse than no test because it reads as coverage. The shim keeps the
# pytest decorators meaningful for CI and lets `venv/bin/python tests/test_beam_cube.py` run
# the same assertions today.
try:
    import pytest
except ImportError:                                                    # pragma: no cover
    class _Approx:
        def __init__(self, v, abs=0.0):
            self.v, self.abs = v, abs

        def __eq__(self, other):
            return abs(other - self.v) <= (self.abs or 1e-12)

    class _Pytest:
        @staticmethod
        def approx(v, abs=0.0):
            return _Approx(v, abs)

        class mark:
            @staticmethod
            def parametrize(_names, values):
                def deco(fn):
                    fn._params = values
                    return fn
                return deco

        @staticmethod
        def importorskip(name, reason=""):
            return __import__(name)

    pytest = _Pytest()

hp = pytest.importorskip("healpy", reason="healpy lives in /home/kvand/gnss/venv, not venv-ft")

from gnss_beam_cube import (  # noqa: E402
    ang2pix_ring, azel_to_pix, nest2ring, ring2nest, ring_downgrade, angsep_deg)

NSIDES = [4, 8, 16, 32, 64, 128]
N = 20000


def _rng():
    return np.random.default_rng(20260903)


@pytest.mark.parametrize("nside", NSIDES)
def test_ang2pix_ring_matches_healpy(nside):
    r = _rng()
    # Uniform on the sphere, so the polar caps and the equatorial belt are both exercised --
    # they are separate branches, and a bug in one hides behind the other under az/el sampling
    # that never leaves the upper hemisphere.
    theta = np.arccos(r.uniform(-1.0, 1.0, N))
    phi = r.uniform(0.0, 2.0 * np.pi, N)
    assert np.array_equal(ang2pix_ring(nside, theta, phi),
                          hp.ang2pix(nside, theta, phi, nest=False))


@pytest.mark.parametrize("nside", NSIDES)
def test_ring_nest_roundtrip_matches_healpy(nside):
    r = _rng()
    npix = 12 * nside * nside
    p = r.integers(0, npix, N)
    assert np.array_equal(ring2nest(p, nside), hp.ring2nest(nside, p))
    assert np.array_equal(nest2ring(p, nside), hp.nest2ring(nside, p))
    # And they invert each other, which the two comparisons above do not by themselves imply.
    assert np.array_equal(nest2ring(ring2nest(p, nside), nside), p)


@pytest.mark.parametrize("pair", [(64, 16), (64, 32), (32, 8), (128, 16)])
def test_downgrade_equals_reprojection(pair):
    """A downgrade must equal re-pixelising the pixel's own centre at the coarser nside.

    This is the property the export relies on when it coalesces duplicate pixels by SUMMING
    accumulators -- if the mapping were not exactly this, the export would be adding cells
    that do not belong to the same coarse pixel.
    """
    nin, nout = pair
    r = _rng()
    p = r.integers(0, 12 * nin * nin, 5000)
    assert np.array_equal(ring_downgrade(p, nin, nout),
                          hp.ang2pix(nout, *hp.pix2ang(nin, p)))


def test_downgrade_refuses_nothing_silently():
    """Equal nside in and out is the identity, not a no-op that quietly drops the shift."""
    p = _rng().integers(0, 12 * 32 * 32, 500)
    assert np.array_equal(ring_downgrade(p, 32, 32), p)


def test_azel_convention_is_the_local_horizon_frame():
    """The cube is pixelised in the LOCAL horizon frame: colatitude from zenith, phi = azimuth.

    Deliberately local, not celestial -- the beam is bolted to the dish, so a celestial
    pixelisation would smear a stationary pattern across the sky as earth turns.

    ⚠️ THERE IS NO SINGLE ZENITH PIXEL, and this test asserted that there was. In HEALPix the
    pole is a pixel CORNER where four pixels meet, so the four cardinal azimuths at el 90 land
    in pixels 0, 1, 2 and 3 -- which is what healpy does too, verified here rather than
    assumed. Anything that wants "the value at zenith" must average the top ring, never index
    one pixel; a viewer that picks pixel 0 is reading one quadrant and calling it the centre.
    """
    az = np.array([0.0, 90.0, 180.0, 270.0])
    z = azel_to_pix(32, az, np.full(4, 90.0))
    assert np.array_equal(z, hp.ang2pix(32, np.zeros(4), np.radians(az))), \
        "the pole must be pixelised exactly as healpy does, four pixels and all"
    assert sorted(z.tolist()) == [0, 1, 2, 3], "el 90 must land in the top RING of the cap"
    horizon = azel_to_pix(32, az, np.zeros(4))
    assert len(set(horizon.tolist())) == 4, "the four cardinal horizon points must differ"
    # Colatitude, not latitude: zenith sits in the north polar cap, i.e. at a LOW ring index,
    # and the horizon in the equatorial belt. Inverting the two is the classic sign error, and
    # it produces a map that is upside down but perfectly smooth.
    assert z.max() < horizon.min()
    # Monotone in elevation along one azimuth: zenith -> horizon must not jump around.
    ring = azel_to_pix(32, np.full(5, 180.0), np.array([90.0, 70.0, 50.0, 30.0, 10.0]))
    assert list(ring) == sorted(ring), "descending elevation must give ascending RING index"


def test_boresight_separation_is_the_documented_pointing():
    """angsep_deg against the pointing memo's boresight, which the veto and maps both use.

    docs/CHORD_BEAM_MAPS.md §5: az 180.0, el 81.41 -- dishes 8.59 deg SOUTH of zenith. Guard
    against someone "fixing" it to telescope.dish_coelev_deg, which reads -27.3 and is NOT
    this pointing.
    """
    assert angsep_deg(180.0, 81.41, 180.0, 90.0) == pytest.approx(8.59, abs=0.01)
    assert angsep_deg(180.0, 81.41, 180.0, 81.41) == pytest.approx(0.0, abs=1e-9)
    # Symmetric, and blind to a 360 wrap in azimuth.
    assert angsep_deg(0.0, 30.0, 359.0, 30.0) == pytest.approx(
        angsep_deg(359.0, 30.0, 0.0, 30.0), abs=1e-12)
    assert angsep_deg(370.0, 30.0, 10.0, 30.0) == pytest.approx(0.0, abs=1e-9)


if __name__ == "__main__":
    # Standalone driver: same assertions, no pytest. Exits nonzero on the first failure so it
    # is usable as a gate, not just as a report.
    import traceback
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    n_run = n_fail = 0
    for name, fn in fns:
        cases = getattr(fn, "_params", [None])
        for case in cases:
            n_run += 1
            label = "%s%s" % (name, "" if case is None else "[%s]" % (case,))
            try:
                fn() if case is None else fn(case)
                print("  ok   %s" % label)
            except Exception:
                n_fail += 1
                print("  FAIL %s" % label)
                traceback.print_exc()
    print("%d case(s), %d failure(s)" % (n_run, n_fail))
    sys.exit(1 if n_fail else 0)
