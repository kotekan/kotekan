#!/bin/sh
# Capture everything needed to diagnose a WEDGED kotekan, in one shot.
#
# The 2026-07-25 wedge (node alive but 62/68 threads in futex_do_wait, no frames out, REST
# timing out on every endpoint) was lost because ptrace_scope was 1 and no backtrace could be
# taken; by the time that was fixed the process had to be killed to restore service. Run this
# BEFORE killing anything.
#
#   sudo sysctl -w kernel.yama.ptrace_scope=0     # once per boot, if not already 0
#   diag/wedge_capture.sh [outdir]
#
# Attaching STOPS the process for the duration -- irrelevant when it is already wedged, but do
# not run this against a healthy node without meaning to.
set -u
out=${1:-/tmp/wedge_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$out" || exit 1
pid=$(pgrep -x kotekan | head -1)
if [ -z "$pid" ]; then echo "no kotekan running"; exit 1; fi
echo "capturing pid $pid -> $out"

# 1. THE one that matters: full userspace backtrace of every thread.
if [ "$(cat /proc/sys/kernel/yama/ptrace_scope)" = "0" ]; then
    timeout 180 gdb -p "$pid" -batch -ex "set pagination off" \
        -ex "info threads" -ex "thread apply all bt full" > "$out/backtrace.txt" 2>&1
    echo "  backtrace.txt ($(wc -l < "$out/backtrace.txt") lines)"
else
    echo "  SKIPPED backtrace -- ptrace_scope != 0; sudo sysctl -w kernel.yama.ptrace_scope=0"
fi

# 2. Kernel-side state per thread: what each one is blocked on, no ptrace needed.
for t in /proc/"$pid"/task/*; do
    printf "%s %-24s %s\n" "$(basename "$t")" "$(cat "$t/comm" 2>/dev/null)" \
        "$(cat "$t/wchan" 2>/dev/null)"
done > "$out/threads_wchan.txt"
sort -k3 "$out/threads_wchan.txt" | awk '{print $3}' | uniq -c | sort -rn > "$out/wchan_hist.txt"

# 3. Is it stalled or spinning? (ps %cpu is a LIFETIME average -- do not read it as "now".)
top -b -n2 -d 2 -H -p "$pid" > "$out/top_threads.txt" 2>&1
cat /proc/"$pid"/status > "$out/status.txt" 2>&1
cat /proc/"$pid"/stack > "$out/kstack.txt" 2>&1

# 4. Context: who stopped first, the pipeline or the REST server?
tail -400 /tmp/gps_3band.log > "$out/kotekan_tail.log" 2>&1
for b in /tmp/gps_l1_broker.log /tmp/gps_l1_broker_gal.log /tmp/gps_l1_broker_bds.log \
         /tmp/gps_l1_broker_l1c.log; do
    [ -f "$b" ] && tail -200 "$b" > "$out/$(basename "$b")" 2>&1
done
{
    echo "== date"; date
    echo "== uptime/etime"; ps -o pid,stat,etime,%cpu,rss,nlwp -p "$pid"
    echo "== last obs write"; ls -la --time-style=+%H:%M:%S /tmp/gpswipe/obs_*.jsonl
    echo "== REST probe"; for u in l1_gps_combiner l1_gal_combiner l1_gps_track; do
        printf "%-20s " "$u"
        timeout 8 curl -s -o /dev/null -w '%{http_code} %{time_total}s\n' \
            "http://localhost:12048/$u/get_status" || echo TIMEOUT
    done
    echo "== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>&1 | head -3
    echo "== disk"; df -h /
} > "$out/context.txt" 2>&1

echo "done. Start with: $out/wchan_hist.txt then $out/backtrace.txt"
