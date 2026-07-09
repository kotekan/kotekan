#!/usr/bin/env python3
"""fake_aro_stream.py -- synthetic ARO ``networkPowerStream`` producer.

Stands in for the ARO X-engine (``dpdkCore`` -> ``computeDualpolPower`` ->
``networkPowerStream``) so the browser viewer can be developed and tested
offline, with no iceboard, no DPDK, and no squirrel. It speaks the exact
wire protocol ``livebeam_server.py`` expects, but reproduces the parts of
ARO data that differ from the airspy pipeline the viewer was built for:

  * **dual-pol** -- ``num_elems == 2``, Stokes ids ``-5`` (XX) and ``-6``
    (YY), rather than airspy autocorr (1) / crosscorr (4).
  * **uint32 power** -- ``computeDualpolPower`` emits *integer* integrated
    power sums, not the float32 the airspy path (``simpleAutocorr``) sends.
    Pass ``--dtype uint32`` (the default) to match ARO; ``float32`` mimics
    airspy.
  * **inverted, wide band** -- ``freq 600 MHz``, ``sample_bw -400 MHz``
    (channel 0 on top), so ``raw_cadence`` is *negative* and the info-block
    frequency edges descend. This is what exercises the viewer's wide-band
    axis and the abs()-for-timing fix.

Injected structure so every viewer feature has something to chew on:

  * a smooth **bandpass** shape + edge roll-off (for baseline subtraction),
  * a few **persistent narrowband RFI** channels,
  * a **periodic broadband pulse** (a fake pulsar) at ``--pulsar-period``,
    optionally dispersed by ``--dm`` (sweeps later across the band), so the
    fold view has a real signal to fold.

``livebeam_server.py`` *listens*; kotekan *connects*. So this producer is a
TCP client: start livebeam_server first, then run this.

Example
-------
    # terminal 1
    python livebeam_server.py --power-dtype uint32
    # terminal 2
    python tools/fake_aro_stream.py --pulsar-period 1.0 --dm 20
"""

import argparse
import socket
import struct
import sys
import time

import numpy as np

# Must match livebeam_server.KotekanPowerStream.
HEADER_FMT = "=iiiidiiiId"  # IntensityHeader (48 bytes)
SUBFRAME_HDR_FMT = "III"  # IntensityPacketHeader: frame_idx, elem_idx, samples_summed

# Stokes id per pol, matching networkPowerStream's ``-5 - e`` convention
# (-5 = XX, -6 = YY) and the viewer's STOKES_LOOKUP.
STOKES_IDS = [-5, -6, -7, -8]

# Dispersion constant (s): delay = K * DM * (f_GHz^-2 - f_ref_GHz^-2).
DM_CONST_S = 4.148808e-3 * 1e9 / 1e9  # 4.148808e-3 s for DM in pc/cc, f in GHz


def build_header(nfreq, nelem, sample_type, raw_cadence, int_len, utc0):
    return struct.pack(
        HEADER_FMT,
        nfreq * 4,  # packet_length (bytes of spectrum per subframe)
        struct.calcsize(SUBFRAME_HDR_FMT),  # header_length
        nfreq,  # samples_per_packet
        sample_type,  # sample_type (4 = uint32, as networkPowerStream hardcodes)
        raw_cadence,  # raw_cadence (seconds per raw sample; negative if bw < 0)
        nfreq,  # num_freqs
        nelem,  # num_elems
        int_len,  # samples_summed
        0,  # handshake_idx
        utc0,  # handshake_utc
    )


def build_info(nfreq, nelem, freq0_hz, bw_hz):
    """freqs * (lo, hi) float32 edges + nelem int8 Stokes ids.

    Edge formula matches networkPowerStream exactly; with bw < 0 the edges
    descend (channel 0 is the top of the band).
    """
    f = np.arange(nfreq, dtype=np.float64)
    lo = freq0_hz - bw_hz / 2.0 + bw_hz / nfreq * f
    hi = freq0_hz - bw_hz / 2.0 + bw_hz / nfreq * (f + 1.0)
    edges = np.empty(nfreq * 2, dtype=np.float32)
    edges[0::2] = lo
    edges[1::2] = hi
    stokes = np.array(STOKES_IDS[:nelem], dtype=np.int8)
    return edges.tobytes() + stokes.tobytes()


def bandpass(nfreq):
    """A smooth per-channel gain envelope with edge roll-off + gentle ripple."""
    x = np.linspace(0.0, 1.0, nfreq)
    # Raised-cosine passband, rolled off at the edges, plus slow ripple.
    shape = 0.5 - 0.5 * np.cos(2 * np.pi * np.clip((x - 0.05) / 0.90, 0, 1))
    shape = 0.2 + 0.8 * shape
    shape *= 1.0 + 0.10 * np.sin(2 * np.pi * 6 * x)
    return shape.astype(np.float64)


def make_synthesizer(args):
    nfreq = args.nfreq
    bp = bandpass(nfreq)
    base_level = args.level  # mean power-per-sample in the passband

    # Persistent narrowband RFI at a handful of channels.
    rng = np.random.default_rng(1234)
    rfi_chans = rng.choice(nfreq, size=args.rfi_lines, replace=False)
    rfi_gain = np.ones(nfreq)
    rfi_gain[rfi_chans] = rng.uniform(30, 200, size=args.rfi_lines)

    # Channel centre frequencies (GHz) for dispersion delay.
    fedge = args.freq - args.bw / 2.0  # MHz at channel 0 edge
    fcen_ghz = (fedge + args.bw / nfreq * (np.arange(nfreq) + 0.5)) / 1e3

    # Reference frequency for dedispersion delay = top of band.
    f_ref_ghz = fcen_ghz.max()
    disp_delay_s = DM_CONST_S * args.dm * (fcen_ghz ** -2 - f_ref_ghz ** -2)

    def pulse_amp(t_s):
        """Gaussian pulse in pulse phase, per channel (dispersion-delayed)."""
        if args.pulsar_period <= 0:
            return 0.0
        phase = (((t_s - disp_delay_s) / args.pulsar_period) + 0.5) % 1.0
        return args.pulse_snr * np.exp(-0.5 * ((phase - 0.5) / args.pulse_width) ** 2)

    def synth(frame_idx, elem, sec_per_frame):
        t_s = frame_idx * sec_per_frame
        # Slight per-pol gain offset so XX/YY are visibly distinct.
        pol_gain = 1.0 if elem == 0 else 0.85
        mean = base_level * bp * rfi_gain * pol_gain
        # Radiometer-ish per-channel fluctuation.
        mean = mean * (1.0 + rng.normal(0.0, 0.03, size=nfreq))
        # Broadband pulse riding on the passband.
        mean = mean + base_level * bp * pol_gain * pulse_amp(t_s)
        return np.clip(mean, 0.0, None)

    return synth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=23401,
                    help="livebeam_server --kotekan-port (default 23401)")
    ap.add_argument("--nfreq", type=int, default=1024)
    ap.add_argument("--nelem", type=int, default=2, help="pols (2 = dual-pol XX/YY)")
    ap.add_argument("--freq", type=float, default=600.0, help="band centre (MHz)")
    ap.add_argument("--bw", type=float, default=-400.0,
                    help="sample bandwidth (MHz); negative = channel 0 on top (ARO)")
    ap.add_argument("--int-len", type=int, default=2048,
                    help="samples_summed per integration (power_integration_length)")
    ap.add_argument("--dtype", choices=["uint32", "float32"], default="uint32",
                    help="payload dtype: uint32 = ARO computeDualpolPower, "
                         "float32 = airspy simpleAutocorr")
    ap.add_argument("--level", type=float, default=5.0,
                    help="mean power-per-sample in the passband (arb units); "
                         "default ~matches the live ARO 46m level (~7 dB median)")
    ap.add_argument("--rfi-lines", type=int, default=5)
    ap.add_argument("--pulsar-period", type=float, default=1.0,
                    help="fake pulsar spin period (s); 0 disables the pulse")
    ap.add_argument("--pulse-width", type=float, default=0.03,
                    help="pulse width as a fraction of a turn")
    ap.add_argument("--pulse-snr", type=float, default=1.5,
                    help="pulse amplitude relative to the passband level")
    ap.add_argument("--dm", type=float, default=0.0,
                    help="dispersion measure (pc/cm^3) for the pulse sweep")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="wall-clock speed multiplier (>1 = faster than real time)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N integrations (0 = run forever)")
    args = ap.parse_args(argv)

    nfreq, nelem = args.nfreq, args.nelem
    bw_hz = args.bw * 1e6
    freq0_hz = args.freq * 1e6
    raw_cadence = 1.0 / (bw_hz / nfreq)  # negative when bw < 0, as ARO sends
    sec_per_frame = abs(raw_cadence) * args.int_len
    sample_type = 4 if args.dtype == "uint32" else 4  # header tag; both say 4
    np_dtype = np.uint32 if args.dtype == "uint32" else np.float32

    synth = make_synthesizer(args)

    print(f"[fake_aro] connecting to {args.host}:{args.port} ...", file=sys.stderr)
    sock = socket.create_connection((args.host, args.port))
    utc0 = time.time()
    sock.sendall(build_header(nfreq, nelem, sample_type, raw_cadence,
                              args.int_len, utc0))
    sock.sendall(build_info(nfreq, nelem, freq0_hz, bw_hz))
    print(f"[fake_aro] handshake sent: nfreq={nfreq} nelem={nelem} "
          f"dtype={args.dtype} raw_cadence={raw_cadence:.3e}s "
          f"frame={sec_per_frame*1e3:.2f}ms band={args.freq}±{abs(args.bw)/2}MHz",
          file=sys.stderr)

    frame_idx = 0
    t_next = time.monotonic()
    try:
        while True:
            for elem in range(nelem):
                mean = synth(frame_idx, elem, sec_per_frame)
                if np_dtype is np.uint32:
                    payload = np.clip(mean * args.int_len, 0, 2**32 - 1).astype(np.uint32)
                else:
                    payload = mean.astype(np.float32)
                hdr = struct.pack(SUBFRAME_HDR_FMT, frame_idx, elem, args.int_len)
                sock.sendall(hdr + payload.tobytes())
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
            t_next += sec_per_frame / max(args.rate, 1e-6)
            dt = t_next - time.monotonic()
            if dt > 0:
                time.sleep(dt)
    except (BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
        print(f"[fake_aro] stopped after {frame_idx} integrations", file=sys.stderr)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
