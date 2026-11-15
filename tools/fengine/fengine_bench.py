#!/usr/bin/env python3
"""Sweep cudaSparkFEngine over FFT sizes and implementations; sample the per-window GPU
seconds gauge and the window rate from the REST metrics. Appends rows to a CSV."""

import argparse, csv, os, signal, subprocess, sys, time, urllib.request

TEMPLATE = """---
type: config
log_level: INFO
cpu_affinity: [2, 3, 4]
telescope:
    name: fake
header_size: 64
header_slot: 128
payload_size: 6144
payload_slot: 6144
packets_per_frame: 1024
samples_per_packet: 4096
fft_length: {n}
queue_depth: 8192
header_buffer:
    kotekan_buffer: standard
    num_frames: 96
    frame_size: packets_per_frame * header_slot
payload_buffer:
    kotekan_buffer: standard
    num_frames: 96
    frame_size: packets_per_frame * payload_slot
job_buffer:
    kotekan_buffer: standard
    num_frames: 4
    frame_size: 48 + 8 * fft_length / samples_per_packet
noise_buffer:
    kotekan_buffer: standard
    num_frames: 1
    frame_size: 64 * payload_slot
out_payload_buffer:
    kotekan_buffer: standard
    num_frames: 4
    frame_size: fft_length / 2
gen:
    kotekan_stage: crsPacketGen
    cpu_affinity: [2]
    header_buf: header_buffer
    payload_buf: payload_buffer
    src_mac: "30:c5:99:3f:09:1e"
    dst_mac: "10:70:fd:53:05:06"
    src_ip: 10.0.0.1
    dst_ip: 10.0.0.2
    src_port: 4000
    num_adcs: 1
    tone_bin: [5]
    tone_amplitude: [1000]
    noise_rms: 0
    num_frames: -1
align:
    kotekan_stage: crsWindowAlign
    cpu_affinity: [3]
    header_buf: header_buffer
    payload_buf: payload_buffer
    job_buf: job_buffer
    noise_buf: noise_buffer
    noise_slots: 64
fengine:
    kotekan_stage: cudaSparkFEngine
    cpu_affinity: [4]
    job_buf: job_buffer
    payload_buf: payload_buffer
    noise_buf: noise_buffer
    out_payload_buf: out_payload_buffer
    gain: 8
    fft_precision: {precision}
    fft_impl: {impl}
"""


def scrape(port):
    with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5) as r:
        txt = r.read().decode()
    vals = {}
    for line in txt.splitlines():
        if line.startswith("kotekan_cuda_spark_fengine_gpu_seconds"):
            vals["gpu_s"] = float(line.split("}")[1].split()[0])
        if line.startswith("kotekan_cuda_spark_fengine_windows_total"):
            vals["windows"] = float(line.split("}")[1].split()[0])
    return vals


def run_one(binary, n, precision, impl, samples, scratch):
    cfg = os.path.join(scratch, f"bench_{n}_{precision}_{impl}.yaml")
    with open(cfg, "w") as f:
        f.write(TEMPLATE.format(n=n, precision=precision, impl=impl))
    log = cfg.replace(".yaml", ".log")
    proc = subprocess.Popen(
        [binary, "-c", cfg], stdout=open(log, "w"), stderr=subprocess.STDOUT
    )
    try:
        # Wait for the gauge to appear (warmup, alignment, first windows).
        t0 = time.time()
        while time.time() - t0 < 60:
            time.sleep(1.0)
            if proc.poll() is not None:
                raise RuntimeError(f"kotekan exited early; see {log}")
            try:
                if "gpu_s" in scrape(12048):
                    break
            except Exception:
                pass
        else:
            raise RuntimeError(f"no gpu_seconds within 60 s; see {log}")
        vals = []
        w0 = scrape(12048).get("windows", 0)
        tw0 = time.time()
        for _ in range(samples):
            time.sleep(0.3)
            v = scrape(12048).get("gpu_s")
            if v:
                vals.append(v)
        w1 = scrape(12048).get("windows", 0)
        tw1 = time.time()
        wps = (w1 - w0) / (tw1 - tw0)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    period = n / 3.2e9
    mean = sum(vals) / len(vals)
    return {
        "n": n,
        "precision": precision,
        "impl": impl,
        "gpu_s_mean": mean,
        "gpu_s_min": min(vals),
        "gpu_s_max": max(vals),
        "samples": len(vals),
        "windows_per_s": wps,
        "window_period_s": period,
        "utilisation": mean / period,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/kotekan/kotekan")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--impl", default="cufft")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--sizes", default="20,21,22,23,24,25,26,27")
    ap.add_argument("--scratch", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    rows = []
    for e in [int(x) for x in a.sizes.split(",")]:
        r = run_one(a.binary, 1 << e, a.precision, a.impl, a.samples, a.scratch)
        print(
            f"2^{e} {a.precision}/{a.impl}: {r['gpu_s_mean']*1e3:.3f} ms/window "
            f"({r['utilisation']*100:.1f}% of realtime), {r['windows_per_s']:.1f} w/s",
            flush=True,
        )
        rows.append(r)
        time.sleep(2)
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
