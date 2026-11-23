"""A live site move: Receiver.site_target, the publisher's POST /set_site and gnss_set_site's yaml
rewrite. Dummy positions only.

    python3 -m unittest test_site_move
"""

import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request

import numpy as np

import gnss_pvt
import gnss_set_site
from gnss_broker import publish
from gnss_broker.receiver import Receiver


def enu(a, b):
    """b - a, east/north/up metres, both (lat, lon, alt)."""
    return gnss_pvt._enu_axes(a[0], a[1]) @ (
        gnss_pvt._llh_to_ecef(*b) - gnss_pvt._llh_to_ecef(*a)
    )


class TargetTest(unittest.TestCase):
    def test_target_is_the_offset_and_shared(self):
        rx = Receiver(log=lambda m: None)
        seq = rx.request_site_offset(3.0, -4.0, 12.0)
        t1 = rx.site_target(seq, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(
            enu((0.0, 0.0, 0.0), t1), [3.0, -4.0, 12.0], atol=1e-6
        )
        self.assertEqual(
            rx.site_target(seq, 0.1, 0.1, 50.0), t1, "every chain lands on one point"
        )
        rx.request_site_offset(1.0, 0.0, 0.0)
        self.assertIsNone(rx.site_target(seq, 0.0, 0.0, 0.0), "a superseded request")


class PostTest(unittest.TestCase):
    def setUp(self):
        self.pub = publish.FleetPublisher(0, lambda m: None)
        self.pub.rx = Receiver(log=lambda m: None)
        self.url = "http://127.0.0.1:%d" % self.pub._srv.server_address[1]

    def tearDown(self):
        self.pub._srv.shutdown()

    def post(self, path, body, ctype="application/json"):
        req = urllib.request.Request(
            self.url + path, data=body, headers={"Content-Type": ctype}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def test_post(self):
        code, out = self.post(
            "/set_site", json.dumps({"d_e": 1, "d_n": 2, "d_u": -3}).encode()
        )
        self.assertEqual((code, out["seq"]), (200, 1))
        self.assertEqual(self.pub.rx.site_req["d_enu"], (1.0, 2.0, -3.0))
        self.assertEqual(
            self.post("/set_site", b'{"d_e":1,"d_n":0,"d_u":0}', "text/plain")[0], 415
        )
        self.assertEqual(self.post("/set_site", b'{"d_e":9,"d_n":0,"d_u":0}')[0], 400)
        self.assertEqual(self.post("/set_site", b'{"d_e":1}')[0], 400)
        self.assertEqual(self.post("/get_status", b"{}")[0], 404)
        self.assertEqual(self.pub.rx.site_req["seq"], 1, "refused posts move nothing")


class YamlTest(unittest.TestCase):
    def test_rewrite(self):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write("common:\n    lat: 10.000000\n    lon: 20.000000\n    alt: 100.0\n")
        try:
            got = gnss_set_site.move_yaml(path, [1.0, 2.0, 3.0])
            np.testing.assert_allclose(got, [1.0, 2.0, 3.0], atol=0.1)
            self.assertIn("    alt: 103.0\n", open(path).read())
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
