#!/usr/bin/env python3
"""Read path-B gathered N^2 tiles and report what the synthetic lanes measured.

Layout (must match cudaCorrelatorDual::build_tile_selection):
  frame = [nt_outer=1][n_gnss_chan][n_tile][16][16][2] int32
  per channel: 16 mixed tiles (ihi 8..15) x (jhi 0..1), jhi fastest;
               then the BB triangle (ihi 8..15, jhi 8..ihi), jhi fastest.
  lane L (= 4*prn_slot + {E,P,L,PH}) is global station 128+L.
  V_ij = E_i conj(E_j), i = synthetic row  =>  despread amplitude = conj(V)/M2_diag.

usage: n2tiles.py [glob] [--prns 1,2,..]   (default: the newest gnss0 file in /tmp/gnss)
"""
import glob
import sys

import numpy as np

N_SYNTH, N_LIVE, N_CHAN = 128, 32, 7
NA16 = 128 // 16          # antenna tile rows before the synthetic block
NSY16 = N_SYNTH // 16     # 8 synthetic tile rows
NLIVE16 = (N_LIVE + 15) // 16
N_MIXED = NSY16 * NLIVE16
N_BB = NSY16 * (NSY16 + 1) // 2
N_TILE = N_MIXED + N_BB
FRAME = N_CHAN * N_TILE * 512 * 4

pat = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') \
    else '/tmp/gnss/*gnss0_n2tiles*.raw'
files = sorted(glob.glob(pat))
if not files:
    sys.exit(f'no files matching {pat}')


def load(path):
    raw = open(path, 'rb').read()
    if len(raw) < FRAME:
        return None
    # rawFileWrite prepends a metadata header; the frame is the trailing FRAME bytes.
    return np.frombuffer(raw[-FRAME:], dtype=np.int32).reshape(N_CHAN, N_TILE, 16, 16, 2)


def mixed(frame, lane, chan):
    """[32] complex: this lane against every live antenna, one channel."""
    ihi, ilo = lane // 16, lane % 16
    out = np.zeros(N_LIVE, dtype=complex)
    for e in range(N_LIVE):
        t = ihi * NLIVE16 + (e // 16)          # mixed tiles, jhi fastest
        v = frame[chan, t, ilo, e % 16]
        out[e] = complex(v[0], v[1])
    return out


def bb_diag(frame, lane, chan):
    """This lane's own energy (the M^2 diagonal): real by construction."""
    ihi, ilo = lane // 16, lane % 16
    t = N_MIXED + ihi * (ihi + 1) // 2 + ihi   # BB triangle, jhi fastest
    v = frame[chan, t, ilo, ilo]
    return complex(v[0], v[1])


print(f'{len(files)} files; frame {FRAME} B = {N_CHAN} chan x {N_TILE} tiles '
      f'({N_MIXED} mixed + {N_BB} BB)')
frame = load(files[-1])
if frame is None:
    sys.exit('last file short; still being written?')
print(f'analysing {files[-1]}\n')

# Which slots are live? A slot with any nonzero BB diagonal was synthesized this frame.
live = []
for slot in range(N_SYNTH // 4):
    e = sum(abs(bb_diag(frame, 4 * slot + t, c)) for t in range(4) for c in range(N_CHAN))
    if e > 0:
        live.append(slot)
print(f'live PRN slots this frame: {len(live)} -> {[s + 1 for s in live]}\n')

print(f'{"PRN":>4} {"chan":>4} {"|P|/rms(E,L)":>13} {"E:P:L":>20} '
      f'{"|sum_ant P|":>12} {"coh":>6}')
for slot in live[:8]:
    for c in (3,):  # mid channel
        amps = []
        for t in range(4):
            v = mixed(frame, 4 * slot + t, c)
            d = bb_diag(frame, 4 * slot + t, c).real
            amps.append(np.conj(v) / d if d else np.zeros(N_LIVE, dtype=complex))
        E, P, L, PH = (np.abs(a).mean() for a in amps)
        ratio = P / (0.5 * (E + L)) if (E + L) else 0.0
        coherent = abs(amps[1].sum())
        incoherent = np.abs(amps[1]).sum()
        coh = coherent / incoherent if incoherent else 0.0
        print(f'{slot + 1:>4} {c:>4} {ratio:>13.3f} '
              f'{E:>6.4f}:{P:>6.4f}:{L:>6.4f} {coherent:>12.5f} {coh:>6.3f}')

print('\n  |P|/rms(E,L) > 1 => the prompt beats the +-0.5-chip taps (a real correlation peak)')
print('  coh = |sum_ant| / sum|ant| : 1.0 = all antennas in phase, ~1/sqrt(32)=0.18 = noise')
