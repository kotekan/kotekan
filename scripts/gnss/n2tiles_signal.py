#!/usr/bin/env python3
"""Is there SIGNAL in the path-B N^2 tiles? A self-contained test.

Do NOT use |P|/rms(E,L): on CHORD each PFB bin is ~195 kHz against a 10.23 Mcps code, so the
E/P/L replicas are ~0.97 coherent within one channel and E≈P≈L holds for strong signal too
(see the M^2 block; memory: chord-epl-equal-is-not-noise).

Instead use CROSS-FRAME ELEMENT-VECTOR COHERENCE. For a real satellite the 32-antenna complex
vector is set by array geometry and is stable frame to frame; any common phase rotation (Doppler,
clock) cancels in a NORMALIZED inner product. Noise re-randomises every frame.

    coh = |<v_k, v_{k+1}>| / (|v_k| |v_{k+1}|)      signal -> ~1,  noise -> ~1/sqrt(32) = 0.18

Reported per lane (E/P/L/PH) per PRN slot, averaged over consecutive frame pairs and channels.

usage: n2tiles_signal.py [glob] [n_frames]
"""
import glob
import os
import sys

import numpy as np

N_SYNTH, N_LIVE, N_CHAN = 128, 32, 7
NSY16, NLIVE16 = N_SYNTH // 16, (N_LIVE + 15) // 16
N_MIXED = NSY16 * NLIVE16
N_BB = NSY16 * (NSY16 + 1) // 2
N_TILE = N_MIXED + N_BB
FRAME = N_CHAN * N_TILE * 512 * 4

pat = sys.argv[1] if len(sys.argv) > 1 else '/tmp/gnss/*gnss0_n2tiles*.raw'
nwant = int(sys.argv[2]) if len(sys.argv) > 2 else 12
files = sorted(glob.glob(pat), key=os.path.getmtime)[-nwant:]
if len(files) < 2:
    sys.exit('need >= 2 files')


def load(p):
    raw = open(p, 'rb').read()
    if len(raw) < FRAME:
        return None
    return np.frombuffer(raw[-FRAME:], dtype=np.int32).reshape(N_CHAN, N_TILE, 16, 16, 2)


def mixed_vec(fr, lane, chan):
    """[32] complex: lane vs every live antenna."""
    ihi, ilo = lane // 16, lane % 16
    out = np.empty(N_LIVE, dtype=complex)
    for e in range(N_LIVE):
        t = ihi * NLIVE16 + (e // 16)
        v = fr[chan, t, ilo, e % 16]
        out[e] = complex(v[0], v[1])
    return out


def bb_diag(fr, lane, chan):
    ihi, ilo = lane // 16, lane % 16
    t = N_MIXED + ihi * (ihi + 1) // 2 + ihi
    return float(fr[chan, t, ilo, ilo][0])


frames = [f for f in (load(p) for p in files) if f is not None]
print(f'{len(frames)} frames from {os.path.basename(files[0])} .. {os.path.basename(files[-1])}')

live = [s for s in range(N_SYNTH // 4)
        if sum(abs(bb_diag(frames[-1], 4 * s + t, c)) for t in range(4) for c in range(N_CHAN)) > 0]
print(f'live PRN slots: {[s + 1 for s in live]}\n')
print(f'{"PRN":>4} {"lane":>5} {"cross-frame coh":>16} {"|v| med":>12}')
print(f'{"":>4} {"":>5} {"(noise ~0.18)":>16}')
for slot in live:
    for t, nm in enumerate(('E', 'P', 'L', 'PH')):
        lane = 4 * slot + t
        cohs, mags = [], []
        for c in range(N_CHAN):
            vs = [mixed_vec(f, lane, c) for f in frames]
            for a, b in zip(vs[:-1], vs[1:]):
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0 and nb > 0:
                    cohs.append(abs(np.vdot(a, b)) / (na * nb))
            mags += [np.linalg.norm(v) for v in vs]
        if cohs:
            flag = '  <== SIGNAL' if np.mean(cohs) > 0.4 else ''
            print(f'{slot + 1:>4} {nm:>5} {np.mean(cohs):>16.3f} {np.median(mags):>12.3e}{flag}')
print('\n  coh ~0.18 = noise; >0.4 = a stable per-antenna phase pattern, i.e. a real source')
