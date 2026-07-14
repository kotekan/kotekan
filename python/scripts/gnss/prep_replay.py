#!/usr/bin/env python3
"""Preprocess a raw airspy_rx L1 capture into /tmp/gpsin for the kotekan offline replay bench
(config/offline_l1ca_replay.yaml). Mimics airspyInput: DC-subtract + (-1)^n nyquist flip, then the
rawFileRead framing = [uint32 metadata_size=0][int16 samples].

Usage:  python3 prep_replay.py [capture.bin]     (default: newest l1_5msps_*.bin here)
"""
import numpy as np, struct, os, sys, glob

here=os.path.dirname(os.path.abspath(__file__))
path=sys.argv[1] if len(sys.argv)>1 else max(glob.glob(os.path.join(here,"l1_5msps_*.bin")),key=os.path.getmtime)
x=np.fromfile(path,dtype=np.int16).astype(np.int32)
x-=int(x.mean())                        # DC-subtract (airspyInput does this)
x[1::2]*=-1                             # (-1)^n flip -> L1 carrier at Fs/4 = 1.25 MHz
os.makedirs("/tmp/gpsin",exist_ok=True)
out="/tmp/gpsin/gpsin_0000000.raw"
with open(out,"wb") as f:
    f.write(struct.pack("<I",0))        # metadata_size = 0
    f.write(x.astype(np.int16).tobytes())
print("prepped %s (%.1f s) -> %s"%(os.path.basename(path),len(x)/5e6,out))
print("replay: cd <repo>/kotekan && ./build_mac/kotekan/kotekan -c config/offline_l1ca_replay.yaml -b 0.0.0.0:12048")
print("        + run python/scripts/gps_distributed_broker.py alongside (blind search seeds the tracker,")
print("        its cp_rate slope fit is what absorbs the LO-vs-ADC code-rate offset)")
