#!/usr/bin/env python3
"""Direct time-domain correlation C/N0 on a raw 20 MSPS airspy grab (grab_l1.sh style,
airspy_rx -t 3 -a 0) -- INDEPENDENT of the kotekan channelizer/replica/DLL, with code
generators written from scratch (C/A shift registers, B1C Weil, E1C parsed from the repo
hex table). This is the absolute-scale ground truth that splits pipeline-loss from
antenna-loss: on 2026-07-12 it showed B1C/E1C at full strength on the air while the
pipeline sat ~10 dB low, exposing the BOC DLL false lock (power discriminator at +-0.5
COMPONENT chips locks +-0.25 chips off-peak, prompt R(0.25) = 0.25 = -12 dB).

Per block (one primary code period): mix to baseband at f0 = IF + dop, FFT circular
correlation against the full-rate replica, x = (peak - floor)/floor, C/N0 = x/T_p.
Averages x over blocks (peak walk between blocks is irrelevant: per-block peak search).
EDIT the targets list at the bottom for the sats above your horizon (PRN + rough broker
Doppler); GPS PRN 6 doubles as the spectral/Doppler convention calibrator.

Usage: python3 gnss_direct_cn0.py [raw.bin]
"""
import re
import sys
import numpy as np

FS = 20e6
IF = 5e6
FC = 1575.42e6
RAW = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/lwlab/airspy_gps/captures/l1_20msps_2026-07-12.bin"
T_SKIP = 2.0  # skip startup

# ---------------- code generators ----------------
def ca_code(prn):
    taps = {1:(2,6),2:(3,7),3:(4,8),4:(5,9),5:(1,9),6:(2,10),7:(1,8),8:(2,9),9:(3,10),
            10:(2,3),11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),17:(1,4),
            18:(2,5),19:(3,6),20:(4,7),21:(5,8),22:(6,9),23:(1,3),24:(4,6),25:(5,7),
            26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),31:(3,8),32:(4,9)}
    g1 = np.ones(10, int); g2 = np.ones(10, int)
    t1, t2 = taps[prn]
    out = np.empty(1023, int)
    for i in range(1023):
        out[i] = g1[9] ^ (g2[t1-1] ^ g2[t2-1])
        f1 = g1[2] ^ g1[9]
        f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = np.roll(g1, 1); g1[0] = f1
        g2 = np.roll(g2, 1); g2[0] = f2
    return 1 - 2*out  # 0->+1, 1->-1

def legendre(N):
    L = np.ones(N, np.int8)
    i = np.arange(1, N, dtype=np.int64)
    L[(i*i) % N] = -1
    return L

B1CP_PH = [796,156,4198,3941,1374,1338,1833,2521,3175,168,2715,4408,3160,2796,459,3594,
           4813,586,1428,2371,2285,3377,4965,3779,4547,1646,1430,607,2118,4709,1149,3283,
           2473,1006,3670,1817,771,2173,740,1433,2458,3459,2155,1205,413,874,2463,1106,
           1590,3873,4026,4272,3556,128,1200,130,4494,1871,3073,4386,4098,1923,1176]
B1CP_TR = [7575,2369,5688,539,2270,7306,6457,6254,5644,7119,1402,5557,5764,1073,7001,5910,
           10060,2710,1546,6887,1883,5613,5062,1038,10170,6484,1718,2535,1158,526,7331,5844,
           6423,6968,1280,1838,1989,6468,2091,1581,1453,6252,7122,7711,7216,2113,1095,1628,
           1713,6102,6123,6070,1115,8047,6795,2575,53,1729,6388,682,5565,7160,2277]

def b1cp_code(prn):
    L = legendre(10243)
    w, p = B1CP_PH[prn-1], B1CP_TR[prn-1]
    i = np.arange(10230)
    j = (i + p - 1) % 10243
    return (L[j] * L[(j + w) % 10243]).astype(int)

def e1c_code(prn):
    src = open("/home/lwlab/airspy_gps/kotekan/lib/stages/gnss/galileoE1Code.cpp").read()
    m = re.search(r"E1C_HEX\[50\]\s*=\s*\{(.*?)\n\};", src, re.S)
    entries = re.findall(r'((?:"[0-9A-Fa-f]+"\s*)+),', m.group(1) + ",")
    hexstr = "".join(re.findall(r'"([0-9A-Fa-f]+)"', entries[prn-1]))
    bits = np.array([int(c, 16) for c in hexstr], int)
    chips = np.zeros(4092, int)
    for b in range(4):
        chips[b::4] = (bits >> (3-b)) & 1
    return 1 - 2*chips

# ---------------- correlator ----------------
def replica(code, chip_rate, boc, n_samp):
    if boc:
        code = np.repeat(code, 2) * np.tile([1, -1], len(code))
        chip_rate *= 2
    idx = (np.arange(n_samp) * chip_rate / FS).astype(np.int64) % len(code)
    return code[idx].astype(np.float32)

def measure(raw, name, code, chip_rate, T_p, boc, dops, n_blocks, flip):
    n_samp = int(round(FS * T_p))
    rep = replica(code, chip_rate, boc, n_samp)
    REPF = np.conj(np.fft.fft(rep))
    off0 = int(T_SKIP * FS)
    tt = np.arange(n_samp) / FS
    guard = int(2 * FS / chip_rate)  # +-2 primary chips around peak excluded from floor
    best = (-1, 0)
    for dop in dops:
        f0 = IF + dop
        xs = []
        for b in range(n_blocks):
            o = off0 + b * n_samp
            x = raw[o:o+n_samp].astype(np.float32)
            if flip:
                x = x * np.where(np.arange(o, o+n_samp) % 2, -1, 1).astype(np.float32)
            xb = x * np.exp(-2j*np.pi*f0*(tt + o/FS)).astype(np.complex64)
            c = np.abs(np.fft.ifft(np.fft.fft(xb) * REPF))**2
            pk = int(np.argmax(c))
            m = np.ones(n_samp, bool)
            m[max(0,pk-guard):pk+guard] = False
            floor = c[m].mean()
            xs.append((c[pk] - floor) / floor)
        mx = float(np.mean(xs))
        if mx > best[0]:
            best = (mx, dop)
    x, dop = best
    cn0 = 10*np.log10(max(x, 1e-9) / T_p)
    print("%s: best dop %+7.1f Hz  x=%7.2f  C/N0 = %5.1f dB-Hz  (%d blocks of %.0f ms)"
          % (name, dop, x, cn0, n_blocks, T_p*1e3))
    return cn0, dop, x

raw = np.memmap(RAW, dtype=np.int16, mode="r")
print("raw: %.1f s @ 20 MSPS" % (len(raw)/FS))

# --- calibrate conventions on GPS PRN 6 (broker dop ~ -1870..-2120 at grab time) ---
ca6 = ca_code(6)
found = False
for flip in (False, True):
    cn0, dop, x = measure(raw, "G6 cal flip=%d" % flip, ca6, 1.023e6, 1e-3, False,
                          np.arange(-4000, 4001, 250.0), 10, flip)
    if x > 3:
        found = True
        break
print("convention: flip=%s, G6 dop %+.0f" % (flip, dop))

# refine G6 and measure all three targets
targets = [
    ("GPS  PRN 6 (C/A)",  ca6,          1.023e6, 1e-3,  False, dop),
    ("GPS  PRN 11 (C/A)", ca_code(11),  1.023e6, 1e-3,  False, None),
    ("BDS  PRN 21 (B1Cp)", b1cp_code(21), 1.023e6, 10e-3, True,  None),
    ("BDS  PRN 42 (B1Cp)", b1cp_code(42), 1.023e6, 10e-3, True,  None),
    ("GAL  PRN 23 (E1C)", e1c_code(23), 1.023e6, 4e-3,  True,  None),
    ("GAL  PRN 33 (E1C)", e1c_code(33), 1.023e6, 4e-3,  True,  None),
]
for name, code, cr, tp, boc, d0 in targets:
    if d0 is None:
        # coarse: full +-4 kHz at half-bin steps, few blocks
        step = max(0.25/tp/2, 100.0)
        _, d0, _ = measure(raw, name+" coarse", code, cr, tp, boc,
                           np.arange(-4000, 4001, step), 6, flip)
    step = 0.25/tp/4
    measure(raw, name, code, cr, tp, boc, d0 + np.arange(-8, 8.01)*step, 24, flip)
