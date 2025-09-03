"""

This script fragment, plus the accompanying "_post.py", are used to
format the CUDA Baseband beamformer one-hot test results into a
summary table like

VOLTAGE    CPU(re,im)       CPU(re,im)sh     CPU       GPU
0x10      ( -16,    0)      (  -1,    0)     0xf0      0xf0
0x20      (   0,   32)      (   0,    2)     0x02      0x02
0x40      (   0,   64)      (   0,    4)     0x04      0x04
0x80      ( 128,    0)      (   7,    0)     0x70      0x70
0xff      (   0,    0)      (   0,    0)     0x00      0x00

It's used like

./kotekan -c ../../config/tests/onehot_cuda_baseband_beamformer_phase.yaml > log 2>&1 &
(cat ../../tests/bb_bf_onehot_pre.py; cat log | grep PY | sed "s/.*: PY //g"; cat ../../tests/bb_bf_onehot_post.py) > /tmp/run.py
python /tmp/run.py

"""
bb = {}
onehot = {}
sparse = {}
sparse["simulated_formed_beams_buffer"] = {}
sparse["host_formed_beams_buffer"] = {}
