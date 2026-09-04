#!/usr/bin/env python3
"""Preprocess a raw airspy_rx capture into a /tmp replay dir for the kotekan offline bench.
Mimics airspyInput: DC-subtract + (-1)^n nyquist flip (carrier -> Fs/4 = 1.25 MHz, the SAME
front-end model for L1 / L2C / L5 -- all tune the LO on the carrier), then the rawFileRead
framing = [uint32 metadata_size=0][int16 samples]. rawFileRead then streams fixed
samples_per_data_set-sized frames out of that single blob.

Usage:  python3 prep_replay.py [capture.bin] [outdir]
          capture.bin : default = newest l1_5msps_*.bin here
          outdir      : default = /tmp/gpsin   (use /tmp/gpsin_l2c etc. to keep bands separate)
"""
import numpy as np, struct, os, sys, glob

here=os.path.dirname(os.path.abspath(__file__))
path=sys.argv[1] if len(sys.argv)>1 else max(glob.glob(os.path.join(here,"l1_5msps_*.bin")),key=os.path.getmtime)
outdir=sys.argv[2] if len(sys.argv)>2 else "/tmp/gpsin"
x=np.fromfile(path,dtype=np.int16).astype(np.int32)
x-=int(x.mean())                        # DC-subtract (airspyInput does this)
x[1::2]*=-1                             # (-1)^n flip -> carrier at Fs/4 = 1.25 MHz
os.makedirs(outdir,exist_ok=True)
out=os.path.join(outdir,"gpsin_0000000.raw")
with open(out,"wb") as f:
    f.write(struct.pack("<I",0))        # metadata_size = 0
    f.write(x.astype(np.int16).tobytes())
print("prepped %s (%.1f s) -> %s"%(os.path.basename(path),len(x)/5e6,out))
print("replay: cd <repo>/kotekan && ./build_mac/kotekan/kotekan -c config/offline_l1ca_replay.yaml -b 0.0.0.0:12048")
print("        + run python/scripts/gps_distributed_broker.py alongside (blind search seeds the tracker,")
print("        its cp_rate slope fit is what absorbs the LO-vs-ADC code-rate offset)")
