#!/usr/bin/env python3
"""Sweep the airspy LO across a frequency range and poll the ADC rms at each step --
the cheap test for "is the higher rms at L2 (1227.6) vs L1 (1575.42) real in-band RFI,
or just the tuner's conversion-gain slope?"

  * A SMOOTH, monotonic rms(freq) curve  -> GAIN SLOPE (the R820T conversion gain rolls
    with tune frequency). A gain slope scales signal AND noise together, so it does NOT
    hurt C/N0 -- it would mean the L2C weakness is elsewhere (antenna slope, L2C being
    intrinsically ~3 dB weaker, geometry), not a raised noise floor.
  * A localized BUMP / spike at some frequencies -> RFI sitting in the band there. RFI
    raises the floor without raising the signal, so it DOES cost C/N0.

This is a pure REST client: start kotekan yourself with any config that has an airspyInput
stage (airspy_autocorr.yaml is ideal -- you can watch the live spectrum in the viewer while
this sweeps), then run this. It retunes via POST /set_config and reads GET /adcstat, restoring
the original LO at the end.

  # terminal 1: run the autocorr pipeline (also serves the spectrum viewer on :8080)
  ./build_mac/kotekan/kotekan -c config/airspy_autocorr.yaml
  # terminal 2: sweep 1150..1650 MHz and plot rms(freq)
  python3 python/scripts/gps_gain_sweep.py --start 1150 --stop 1650 --step 10

For a fair gain-slope read, keep the gains FIXED across the sweep (this script never touches
them) and keep the SAME input connected (antenna+bias, or a terminator) the whole time. With
the active antenna + bias on you measure the WHOLE chain's slope (tuner + antenna LNA); with a
terminator you isolate the airspy front end.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

GPS_BANDS = {  # MHz, for annotating the plot
    "L5": 1176.45,
    "L2": 1227.60,
    "L1": 1575.42,
}


def _get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def adcstat(base, stage, timeout, navg, settle):
    """Average `navg` adcstat reads (each is one frame's rms) for a tighter estimate."""
    rms = mean = rail = 0.0
    n = 0
    for _ in range(navg):
        try:
            s = _get("%s/%s/adcstat" % (base, stage), timeout)
        except (urllib.error.URLError, OSError, ValueError):
            continue
        rms += float(s.get("rms", 0.0))
        mean += float(s.get("mean", 0.0))
        rail += float(s.get("railfrac", 0.0))
        n += 1
        time.sleep(settle)
    if n == 0:
        return None
    return rms / n, mean / n, rail / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost:12048", help="kotekan REST host:port")
    ap.add_argument("--stage", default="airspy_input",
                    help="airspyInput unique_name (airspy_autocorr.yaml: airspy_input; "
                         "live configs: airspy_in)")
    ap.add_argument("--start", type=float, default=1150.0, help="sweep start, MHz")
    ap.add_argument("--stop", type=float, default=1650.0, help="sweep stop, MHz")
    ap.add_argument("--step", type=float, default=10.0, help="sweep step, MHz")
    ap.add_argument("--dwell", type=float, default=0.3,
                    help="settle time after a retune before reading, s")
    ap.add_argument("--navg", type=int, default=4, help="adcstat reads to average per point")
    ap.add_argument("--timeout", type=float, default=5.0, help="per-request timeout, s")
    ap.add_argument("--csv", default="", help="also write freq,rms,mean,railfrac to this file")
    ap.add_argument("--no-restore", action="store_true",
                    help="leave the LO at the last swept freq (default: restore original)")
    args = ap.parse_args()

    base = "http://" + args.host
    # Remember the starting LO so we can put it back.
    try:
        cfg0 = _get("%s/%s/get_config" % (base, args.stage), args.timeout)
        f0_hz = float(cfg0.get("freq", 0.0))
    except Exception as e:  # noqa: BLE001 -- want a clear message, not a traceback
        sys.exit("Could not reach %s/%s/get_config (%s). Is kotekan running with an "
                 "airspyInput stage named %r?" % (base, args.stage, e, args.stage))
    print("airspy %s @ %s : sweeping %.0f..%.0f MHz step %.0f (LO starts at %.3f MHz)"
          % (args.stage, args.host, args.start, args.stop, args.step, f0_hz / 1e6))

    freqs, rmss = [], []
    rows = []
    f = args.start
    while f <= args.stop + 1e-6:
        try:
            _post("%s/%s/set_config" % (base, args.stage), {"freq": f}, args.timeout)
        except Exception as e:  # noqa: BLE001
            print("  %.1f MHz: set_config failed (%s)" % (f, e))
            f += args.step
            continue
        time.sleep(args.dwell)
        st = adcstat(base, args.stage, args.timeout, args.navg, max(0.02, args.dwell / 4))
        if st is None:
            print("  %.1f MHz: no adcstat" % f)
            f += args.step
            continue
        rms, mean, rail = st
        freqs.append(f)
        rmss.append(rms)
        rows.append((f, rms, mean, rail))
        flag = ""
        if rail > 0.001:
            flag = "  <-- CLIPPING (rail %.3f): lower gains" % rail
        print("  %7.1f MHz   rms %7.1f   mean %7.1f%s" % (f, rms, mean, flag))
        f += args.step

    if not rmss:
        sys.exit("No points collected.")

    # ---- ASCII rms(freq) plot, GPS carriers annotated ----------------------------
    rmin, rmax = min(rmss), max(rmss)
    span = (rmax - rmin) or 1.0
    width = 50
    print("\nrms vs freq  (rms %.1f .. %.1f):" % (rmin, rmax))
    for fr, rm in zip(freqs, rmss):
        bar = int(round((rm - rmin) / span * width))
        tag = ""
        for name, fc in GPS_BANDS.items():
            if abs(fr - fc) <= args.step / 2.0:
                tag = " %s" % name
        print("  %7.1f |%s%s %.0f%s" % (fr, "#" * bar, " " * (width - bar), rm, tag))

    # Slope vs bumpiness summary: fit a line, report the residual scatter. A clean slope
    # has small residual; RFI bumps inflate it.
    n = len(freqs)
    mx = sum(freqs) / n
    my = sum(rmss) / n
    sxx = sum((x - mx) ** 2 for x in freqs) or 1.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(freqs, rmss))
    slope = sxy / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(freqs, rmss)]
    rms_resid = (sum(r * r for r in resid) / n) ** 0.5
    print("\nlinear fit: rms ~ %.3f /MHz (%.0f rms across the %.0f MHz span); "
          "residual scatter %.1f rms" % (slope, slope * (freqs[-1] - freqs[0]),
                                         freqs[-1] - freqs[0], rms_resid))
    print("  small residual + monotonic bars  -> GAIN SLOPE (does NOT hurt C/N0)")
    print("  a bar that juts above the trend   -> RFI at that freq (DOES hurt C/N0)")

    def near(fc):
        return min(zip(freqs, rmss), key=lambda fr: abs(fr[0] - fc))
    if args.start - 1 <= GPS_BANDS["L1"] <= args.stop + 1 and \
       args.start - 1 <= GPS_BANDS["L2"] <= args.stop + 1:
        l1f, l1r = near(GPS_BANDS["L1"])
        l2f, l2r = near(GPS_BANDS["L2"])
        if l1r > 0:
            import math
            print("L2/L1 rms ratio = %.2f (%.1f dB)  [L2 %.0f @ %.0f MHz vs L1 %.0f @ %.0f MHz]"
                  % (l2r / l1r, 20 * math.log10(l2r / l1r), l2r, l2f, l1r, l1f))

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("freq_mhz,rms,mean,railfrac\n")
            for fr, rm, me, ra in rows:
                fh.write("%.3f,%.4f,%.4f,%.6f\n" % (fr, rm, me, ra))
        print("wrote %s" % args.csv)

    if not args.no_restore and f0_hz > 0:
        try:
            _post("%s/%s/set_config" % (base, args.stage), {"freq": f0_hz / 1e6}, args.timeout)
            print("restored LO to %.3f MHz" % (f0_hz / 1e6))
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
