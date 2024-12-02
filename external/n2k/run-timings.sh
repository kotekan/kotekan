#!/bin/bash
set -e
set -x

mkdir -p logs
make -j
rm -f timings.txt

for nt in 262144 16384 8192 4096 2048 1024; do
    ./time-correlator --nt-inner=$nt | tee logs/log_time_$nt
    tail -1 logs/log_time_$nt >>timings.txt
done
