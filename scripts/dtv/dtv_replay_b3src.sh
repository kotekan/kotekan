#!/bin/bash
# Protected-start replay with three source slots (dtv_replay_run.sh recipe B, one more peel),
# scored out of sample on frames 10-19 after each source.
#   dtv_replay_b3src.sh [freqs=2720,2912]
set -e
F=${1:-2720,2912}; N=3
PY=~/gnss/venv/bin/python3; BB=~/gnss/bb1030; ST=~/gnss/dtv-wt/scripts/gnss/dtv_state.py
W6=/var/tmp/bb1030/dtv/replay3; W5=/var/tmp/kvand-dtv/replay3; RP=/var/tmp/kvand-dtv/dtv_replay
gpu() { ssh cf05 "cd $W5 && CUDA_VISIBLE_DEVICES=1 $RP --in x.bin --freq $F --nsrc $N $*"; }
put() { for f in "$@"; do scp -q cf06:$W6/$f cf05:$W5/$f; done; }
get() { for f in "$@"; do scp -q cf05:$W5/$f cf06:$W6/$f; done; }
on6() { ssh cf06 "cd $W6 && nice $PY $*" 2>&1 | grep -v Warning; }
ssh cf06 "mkdir -p $W6 && cd $W6 && nice $PY $BB/dtv_replay.py prep --freq $F --out x.bin" | tail -1
ssh cf05 "mkdir -p $W5"; scp -q cf06:$W6/x.bin cf05:$W5/x.bin
on6 $BB/dtv_replay.py protect --freq $F --frames 0-9 --nsrc $N --out s0w.bin; put s0w.bin
prev=s0w.bin
for k in 0 1 2; do
  gpu --state $prev --out o$k.raw; get o$k.raw
  on6 $ST fit --source $k --state $prev --first 0 --count 10 --out s${k}g.bin o$k.raw; put s${k}g.bin
  gpu --state s${k}g.bin --out r$k.raw --res res$k.bin; get res$k.bin
  on6 $BB/dtv_replay.py eval --freq $F --res res$k.bin --frames 10-19 | sed "s/^/  $((k + 1)) source(s): /"
  if [ $k -lt 2 ]; then
    on6 $BB/dtv_replay.py protect --freq $F --frames 0-9 --source $((k + 1)) --state s${k}g.bin --res res$k.bin --out s$((k + 1))w.bin
    put s$((k + 1))w.bin; prev=s$((k + 1))w.bin
  fi
done
echo "== done"
