#!/usr/bin/env python3
"""MGEX differential code biases (CAS Bias-SINEX): measured per-satellite signal biases.

The broadcast group delay (`gnss_ephemeris.group_delay_s`) is what the satellite says about
its own inter-signal delay, and for BeiDou B2a it says nothing: TGD_B2ap lives in B-CNAV2,
which RINEX 3 does not carry. The IGS/MGEX analysis centres estimate the same quantity from a
global network and publish it daily.

The product cannot supply a constellation-common term. Its header says so: "A zero-mean
constellation condition is applied to separate satellite and receiver biases on a daily
basis." It corrects the per-satellite spread, never a constellation offset.

The datum, per constellation, is the combination the broadcast clock is fitted to. The model
already assumes it, so the correction is (datum - our signal):
    GPS     LNAV  -> L1P/L2P ionosphere-free   (C1W/C2W)
    Galileo F/NAV -> E1/E5a ionosphere-free    (C1C/C5Q);  I/NAV -> E1/E5b (C1C/C7Q)
    BeiDou  D1/D2 -> B3I                       (C6I)
Sign convention: the returned value is added to the broadcast satellite clock, as for
`group_delay_s`, so a signal that leaves the satellite later than the datum gets a negative
correction and the model's predicted code phase moves later with it.

Latency: the rapid product lands ~5 days late and code biases move over weeks to months, so a
file a few days old is fine. fetch_dcb returns nothing older than `max_back_days`, and the
broker warns when the product in use is older than its dcb_max_age_days.
"""
import base64
import gzip
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta

from gnss_brdc_supply import _atomic_write_bytes

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps")

F_L1 = F_E1 = 1575.42e6
F_L2 = 1227.60e6
F_E5A = 1176.45e6
F_E5B = 1207.14e6
GAMMA_L1L2 = (F_L1 / F_L2) ** 2
GAMMA_E1E5A = (F_E1 / F_E5A) ** 2
GAMMA_E1E5B = (F_E1 / F_E5B) ** 2


def _token():
    try:
        with open(os.path.join(CACHE, ".earthdata_token")) as f:
            return f.read().strip() or None
    except Exception:
        return None


_UNSET = object()


def token_expiry_days(tok=_UNSET):
    """Days until the Earthdata token expires, or None if that cannot be read.

    Earthdata issues JWTs, whose middle segment carries an `exp` claim. An expired token stops
    every fetch, and fetch_dcb hides that for up to `max_back_days` by serving the cached
    product, so the expiry date is the early warning.

    Never log the token: this returns a number of days, read from a timestamp claim.
    """
    # Not `tok or _token()`: an explicitly passed empty token must report None, not the expiry
    # of the token on disk. Only an omitted argument reads the token file.
    if tok is _UNSET:
        tok = _token()
    if not tok:
        return None
    parts = tok.split(".")
    if len(parts) != 3:
        return None  # not a JWT: no expiry to read, not an error
    try:
        pad = parts[1] + "=" * (-len(parts[1]) % 4)
        exp = json.loads(base64.urlsafe_b64decode(pad)).get("exp")
        return None if exp is None else (float(exp) - time.time()) / 86400.0
    except Exception:
        return None


def product_age_days(path, when=None):
    """Age of a CAS bias product, from the yyyyddd in its filename. None if unparsable.

    The file's mtime is when we fetched it; the epoch in the name is what the biases describe.
    """
    if not path:
        return None
    m = re.search(r"_(\d{4})(\d{3})0000_", os.path.basename(path))
    if not m:
        return None
    d = datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc) + timedelta(
        days=int(m.group(2)) - 1
    )
    return ((when or datetime.now(timezone.utc)) - d).total_seconds() / 86400.0


def _is_gzip(path):
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def fetch_dcb(when=None, cache_dir=CACHE, max_back_days=14, status=None):
    """Newest available CAS daily bias file, cached locally. None if unreachable.

    Walks back a day at a time from the day before `when`: the rapid product lands ~5 days
    late, so the most recent days 404 as a matter of course.

    An expired token looks like success here for up to `max_back_days`. Each day is looked
    for in the local cache before the network, so once the credential stops working this
    keeps returning the newest file on disk, growing staler, and returns None only after the
    window passes. Pass a `status` dict to see which happened: it is filled with `reason`
    (ok / no-token / auth-rejected / not-gzip / unreachable), `path`, `age_days` (of the
    product, not the file), `served_from_cache`, `http` (the last HTTP error status) and
    `token_expiry_days`. Callers should warn on age, not only on absence.
    """
    st = status if status is not None else {}
    st.update(
        reason="no-token",
        path=None,
        age_days=None,
        http=None,
        token_expiry_days=None,
        served_from_cache=None,
    )
    tok = _token()
    if not tok:
        return None
    st["token_expiry_days"] = token_expiry_days(tok)
    st["reason"] = "unreachable"
    when = when or datetime.now(timezone.utc)
    os.makedirs(cache_dir, exist_ok=True)
    for back in range(1, max_back_days + 1):
        d = when - timedelta(days=back)
        doy = d.timetuple().tm_yday
        name = "CAS0OPSRAP_%04d%03d0000_01D_01D_DCB.BIA.gz" % (d.year, doy)
        local = os.path.join(cache_dir, name)
        if os.path.exists(local) and os.path.getsize(local) > 1000:
            if _is_gzip(local):
                st.update(
                    reason="ok",
                    path=local,
                    served_from_cache=True,
                    age_days=product_age_days(local, when),
                )
                return local
            # Not the product (an error or login page saved under its name): drop it and
            # fetch the day again rather than serve it on every reload.
            try:
                os.unlink(local)
            except OSError:
                pass
        url = "https://cddis.nasa.gov/archive/gnss/products/bias/%04d/%s" % (
            d.year,
            name,
        )
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": "Bearer " + tok}
            )
            with urllib.request.urlopen(req, timeout=45) as r:
                raw = r.read()
            if len(raw) < 1000:
                continue
            if raw[:2] != b"\x1f\x8b":
                # A 200 that is not gzip (a login or error page) is not the product.
                st["reason"] = "not-gzip"
                continue
            # Per-writer temp name: per-chain reload threads can fetch the same day at once.
            _atomic_write_bytes(local, raw)
            st.update(
                reason="ok",
                path=local,
                served_from_cache=False,
                age_days=product_age_days(local, when),
            )
            return local
        except urllib.error.HTTPError as e:
            # 401/403 is the credential, not the calendar: every remaining day will
            # fail the same way, so record it in `reason`.
            st["http"] = e.code
            if e.code in (401, 403):
                st["reason"] = "auth-rejected"
            continue
        except Exception:
            continue
    return None


def parse_dcb(path):
    """Bias-SINEX -> {(sysc, prn): {(obs1, obs2): seconds}} for satellite DSB rows only.

    Station rows are skipped: they are the receiver half of the same separation.
    """
    out = {}
    if not path:
        return out
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        for ln in f:
            if not ln.startswith(" DSB "):
                continue
            fld = ln.split()
            # ' DSB  C217 C01           C2I  C6I  <start> <end> ns  <val> <sd>'
            # A station row (' DSB  G    G   ABPO      C1C  C1W ...') splits to a one-letter
            # PRN field and its site code where OBS1 sits; the PRN and obs-code checks drop it.
            if len(fld) < 10:
                continue
            prn, o1, o2 = fld[2], fld[3], fld[4]
            if len(prn) != 3 or prn[0] not in "GECR" or not prn[1:].isdigit():
                continue
            if len(o1) != 3 or len(o2) != 3 or not o1[0] == "C" or not o2[0] == "C":
                continue
            try:
                val = float(fld[-2])
            except ValueError:
                continue
            if fld[-3] != "ns":
                continue
            out.setdefault((prn[0], int(prn[1:])), {})[(o1, o2)] = val * 1e-9
    return out


def _dsb(tab, a, b):
    """DSB(a,b) = bias_a - bias_b, resolved through the stored direction or its reverse."""
    if (a, b) in tab:
        return tab[(a, b)]
    if (b, a) in tab:
        return -tab[(b, a)]
    return None


def signal_bias_s(dcb, sysc, prn, signal, gal_inav=False):
    """Seconds to add to the broadcast clock for `signal`, from measured DCBs; None if the
    needed pairs are absent (the caller falls back to the broadcast TGD/BGD).

    Covers GPS L5, Galileo E5a/E5b and BeiDou B2a/B2b; any other signal returns None. Every
    case is (datum - our signal), the datum being the combination the broadcast clock is
    fitted to. With DSB(a,b) = b_a - b_b, the ionosphere-free identity behind GPS and Galileo:
        b_IF(1,2) - b_X = DSB(1,X) + DSB(1,2)/(gamma_12 - 1)
    """
    tab = (dcb or {}).get((sysc, prn))
    if not tab:
        return None
    sig = str(signal or "").lower()
    if sysc == "G" and "l5" in sig:
        # datum = L1P/L2P IF (C1W/C2W). Reach C5X (or C5Q) via C1C, the pivot CAS publishes.
        d_1c_5x = _dsb(tab, "C1C", "C5X")
        if d_1c_5x is None:
            d_1c_5x = _dsb(tab, "C1C", "C5Q")
        d_1c_1w = _dsb(tab, "C1C", "C1W")
        d_1w_2w = _dsb(tab, "C1W", "C2W")
        if d_1c_5x is None or d_1c_1w is None or d_1w_2w is None:
            return None
        d_1w_5 = d_1c_5x - d_1c_1w  # b_C1W - b_C5X
        return d_1w_5 + d_1w_2w / (GAMMA_L1L2 - 1.0)
    if sysc == "E":
        # The datum depends on the record type, which the caller knows: F/NAV clocks are
        # E1/E5a iono-free, I/NAV clocks E1/E5b. Both DSBs reach every combination exactly,
        # so a record-type flip changes the datum and the correction together.
        #   b_IF(1,x) = (g_x*b_1 - b_x)/(g_x - 1),  d1x = b_1 - b_x
        #   same-band :  datum - b_x  =  g_x * d1x / (g_x - 1)
        #   cross-band:  datum - b_y  =  d1x / (g_x - 1) + d1y
        d15 = _dsb(tab, "C1C", "C5Q")
        if d15 is None:
            d15 = _dsb(tab, "C1X", "C5X")
        d17 = _dsb(tab, "C1C", "C7Q")
        if d17 is None:
            d17 = _dsb(tab, "C1X", "C7X")
        if "e5a" in sig:
            if not gal_inav:
                return None if d15 is None else GAMMA_E1E5A * d15 / (GAMMA_E1E5A - 1.0)
            if d15 is None or d17 is None:
                return None
            return d17 / (GAMMA_E1E5B - 1.0) + d15
        if "e5b" in sig:
            if gal_inav:
                return None if d17 is None else GAMMA_E1E5B * d17 / (GAMMA_E1E5B - 1.0)
            if d15 is None or d17 is None:
                return None
            return d15 / (GAMMA_E1E5A - 1.0) + d17
        return None
    if sysc == "C":
        # datum = B3I (C6I). B2a is C5P (pilot), C5X or C5D; B2b is C7D or C7Z.
        if "b2a" in sig:
            for one, five in (("C1P", "C5P"), ("C1X", "C5X"), ("C1D", "C5D")):
                d15 = _dsb(tab, one, five)
                d16 = _dsb(tab, one, "C6I")
                if d15 is not None and d16 is not None:
                    return d15 - d16  # = b_C6I - b_B2a
            return None
        if "b2b" in sig:
            for one, seven in (("C1P", "C7D"), ("C1X", "C7D"), ("C1X", "C7Z")):
                d17 = _dsb(tab, one, seven)
                d16 = _dsb(tab, one, "C6I")
                if d17 is not None and d16 is not None:
                    return d17 - d16
            return None
        return None
    return None
