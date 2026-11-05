#!/bin/bash
# Full replay of the DTV subtraction stage on dump chunk 0, two recipes, scored out of sample.
#  A (the live recipe): the stage's own eigen starts (source 0 on frames 0-4, source 1 on the
#    residual of frames 0-4), g fitted on frames 4-9, scored on frames 10-19.
#  B (protected starts, the offline study's): weights orthogonal to the Tau A template, g on frames
#    0-9, scored on 10-19.
# GPU steps on cf05, the dump and the scoring on cf06; the work directory is copied between them.
#   dtv_replay_run.sh [freqs=2512,2720,2912]
set -e
F=${1:-2512,2720,2912}
PY=~/gnss/venv/bin/python3; BB=~/gnss/bb1030; ST=~/gnss/dtv-wt/scripts/gnss/dtv_state.py
W6=/var/tmp/bb1030/dtv/replay; W5=/var/tmp/kvand-dtv/replay; RP=/var/tmp/kvand-dtv/dtv_replay
gpu() { ssh cf05 "mkdir -p $W5 && cd $W5 && CUDA_VISIBLE_DEVICES=1 $RP --in x.bin --freq $F $*"; }
put() { for f in "$@"; do scp -q cf06:$W6/$f cf05:$W5/$f; done; }
get() { for f in "$@"; do scp -q cf05:$W5/$f cf06:$W6/$f; done; }
on6() { ssh cf06 "cd $W6 && nice $PY $*" 2>&1 | grep -v Warning; }
ssh cf05 "mkdir -p $W5"; scp -q cf06:$W6/x3.bin cf05:$W5/x.bin

echo "== A: the stage's own eigen starts"
gpu --start 0 --start-frames 5 --out a1.raw; get a1.raw
on6 $ST fit --source 0 --first 4 --count 6 --out stA1.bin a1.raw; put stA1.bin
gpu --state stA1.bin --start 1 --start-frames 5 --out a2.raw; get a2.raw
on6 $ST fit --source 1 --state stA1.bin --first 4 --count 6 --out stA2.bin a2.raw; put stA2.bin
gpu --state stA1.bin --out a3.raw --res resA1.bin; get resA1.bin
gpu --state stA2.bin --out a4.raw --res resA2.bin; get resA2.bin a4.raw
on6 $ST info a4.raw
on6 $BB/dtv_replay.py eval --freq $F --res resA1.bin --frames 10-19 | sed 's/^/  1 source: /'
on6 $BB/dtv_replay.py eval --freq $F --res resA2.bin --frames 10-19 | sed 's/^/  2 sources: /'

echo "== B: protected starts"
on6 $BB/dtv_replay.py protect --freq $F --frames 0-9 --out stB0.bin; put stB0.bin
gpu --state stB0.bin --out b1.raw; get b1.raw
on6 $ST fit --source 0 --state stB0.bin --first 0 --count 10 --out stB1.bin b1.raw; put stB1.bin
gpu --state stB1.bin --out b2.raw --res resB1.bin; get resB1.bin
on6 $BB/dtv_replay.py protect --freq $F --frames 0-9 --source 1 --state stB1.bin --res resB1.bin --out stB1p.bin; put stB1p.bin
gpu --state stB1p.bin --out b3.raw; get b3.raw
on6 $ST fit --source 1 --state stB1p.bin --first 0 --count 10 --out stB2.bin b3.raw; put stB2.bin
gpu --state stB2.bin --out b4.raw --res resB2.bin; get resB2.bin
on6 $BB/dtv_replay.py eval --freq $F --res resB1.bin --frames 10-19 | sed 's/^/  1 source: /'
on6 $BB/dtv_replay.py eval --freq $F --res resB2.bin --frames 10-19 | sed 's/^/  2 sources: /'
echo "== done"
