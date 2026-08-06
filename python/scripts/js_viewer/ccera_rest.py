#!/usr/bin/env python

import json
import threading
import xmlrpc.client
import xmlrpc.server
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
import datetime
import astropy.coordinates as c
import astropy.units as u

altaz = (0, 0)
location = (45.3491, -76.0413, 95)  # ground elevation
cl = None
timer = None
state = "unknown"

l = c.EarthLocation(lat=location[0], lon=location[1])

from tornado.web import RequestHandler


def update_pointing():
    global altaz, timer
    try:
        altaz = cl.query_both_axes()
        print("[alt,az] = ", altaz)
    except KeyboardInterrupt:
        return
    except Exception as e:
        # Log and still reschedule, so a transient telescope-server error
        # doesn't permanently stop pointing (or crash the first call).
        print("update_pointing: query failed:", e)
    timer = threading.Timer(10.0, update_pointing)
    timer.start()


class get_pointing(RequestHandler):
    def get(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET")
        self.set_header(
            "Access-Control-Allow-Headers", "x-prototype-version,x-requested-with"
        )
        self.set_header("Access-Control-Max-Age", "2520")

        aa = c.SkyCoord(
            alt=altaz[0] * u.deg,
            az=altaz[1] * u.deg,
            obstime=datetime.datetime.now(datetime.timezone.utc),
            frame="altaz",
            location=l,
        )
        radec = aa.transform_to(frame="fk5")
        ra = radec.ra.value
        dec = radec.dec.value
        galpt = aa.transform_to(frame="galactic")
        gl = galpt.l.value
        gb = galpt.b.value

        self.write(
            {"alt": altaz[0], "az": altaz[1], "ra": ra, "dec": dec, "gl": gl, "gb": gb}
        )


class get_position(RequestHandler):
    def get(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET")
        self.set_header(
            "Access-Control-Allow-Headers", "x-prototype-version,x-requested-with"
        )
        self.set_header("Access-Control-Max-Age", "2520")
        self.write({"lat": location[0], "lon": location[1], "el": location[2]})


import time

state_expiry = None
dwell_time = 90  # seconds


def set_state(newstate):
    global state, state_expiry
    state = newstate
    if state.startswith("on source"):
        state_expiry = time.time() + dwell_time
    else:
        state_expiry = None
    return newstate


class get_state(RequestHandler):
    def get(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET")
        self.set_header(
            "Access-Control-Allow-Headers", "x-prototype-version,x-requested-with"
        )
        self.set_header("Access-Control-Max-Age", "2520")
        self.write({"state": state, "expiry": state_expiry, "now": time.time()})


class RPCServerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.localServer = xmlrpc.server.SimpleXMLRPCServer(("0.0.0.0", 9000))
        self.localServer.register_function(set_state)

    def run(self):
        self.localServer.serve_forever()


if __name__ == "__main__":
    cl = xmlrpc.client.ServerProxy("http://172.22.121.35:9090/")
    #    rpc_server = xmlrpc.server.SimpleXMLRPCServer(('localhost',9000))
    #    rpc_server.register_function(set_state)
    rpc_server = RPCServerThread()
    rpc_server.start()
    update_pointing()
    app = Application(
        [
            ("/pointing", get_pointing),
            ("/position", get_position),
            ("/state", get_state),
        ]
    )
    app.listen(3000)
    try:
        #        rpc_server.serve_forever()
        IOLoop.instance().start()
    except KeyboardInterrupt:
        timer.cancel()
