************************
The Spark F-engine demo
************************

A soft-realtime F-engine for a CHORD-like telescope, built on the ibverbs UDP stages and a
single GPU with coherent unified memory (the NVIDIA DGX Spark it was written for). A D-engine
(RFSoC) ships raw ADC samples as CRS packets; this pipeline channelizes them and ships
quantized spectra onward to an X-engine.

Pipeline
========

Per ADC (two by default, 3.2 Gsps of 12-bit two's complement samples, 4096 per packet)::

                 header_buf ──► crsWindowAlign ──► job_buf ──┬─► cudaSparkFEngine ──► out_payload_buf ─┐
    ibvUdpRecv ──┤                                           │         ▲ gain_buf                      ├─► ibvUdpSend
                 payload_buf ────────(read in place)─────────┘   fEngineGains                          │
                                      noise_buf ─────────────┘         ▲ stats_buf  crsHeaderGen ──────┘

* ``ibvUdpRecv`` lands packets in arrival order; it never reorders or aligns.
* ``crsWindowAlign`` reads the CRS seq_numbers and emits one ``fft_window_job`` per
  ``fft_length`` samples, aligned to multiples of ``fft_length``: a lookup table of payload
  slot addresses in time order, with lost packets pointed at a noise buffer. Payload data is
  never copied: the GPU reads the packet payloads and the noise slots in place through the
  table, and releases the frames with the cursors the job carries (``fftWindowJob.hpp``
  documents the protocol and the output spectrum convention).
* ``cudaSparkFEngine`` unpacks the 12-bit samples, FFTs each window (``fft_precision: fp32`` is
  one cuFFT call; ``fp16`` is the production four-step path: two batched fp16 cuFFTs around a
  twiddle/transpose pass), applies the gain table, and quantizes to offset-encoded 4+4-bit
  complex straight into the output payload frames.
* ``fEngineGains`` supplies the per-bin gains: a scalar, a raw float32 file, or auto mode,
  which averages the per-bin power the kernels accumulate and holds every bin's noise at
  ``target_rms`` quantizer counts.
* ``crsHeaderGen`` writes the output packet headers: ``stream_id`` is the packet's index in
  the spectrum, ``source_id.link_id`` the ADC, ``seq_number`` the window index
  (start sample / ``fft_length``), with up to 8192 bins per packet.

Testing
=======

* ``crsPacketGen`` simulates the D-engine (tones, noise, loss injection);
  ``fEngineSimulate`` is the CPU reference implementation (fftw) the GPU is verified
  against; ``fEngineCheck`` verifies emitted spectra, per window in process or per packet
  (``wire_mode``) behind a receive stage.
* ``config/ci-tests/cpu_batch/test_fengine_cpu.yaml`` runs the chain end to end in CI.
  ``config/ci-tests/spark_batch/test_spark_fengine_gpu*.yaml`` run it on the GPU. CI has
  no GPU with unified memory, so run them by hand on a Spark with
  ``config/ci-tests/run_tests.sh``.
* ``config/tests/spark_fengine_loopback.yaml`` runs the whole demo on one host over the NIC
  (a ConnectX hairpins raw packets addressed to its own MAC).
* ``tests/ibverbs/run_fengine.sh`` runs the two-host trio:
  ``crs_dengine_sim.yaml`` (sender) → ``spark_fengine.yaml`` (engine) →
  ``fengine_check.yaml`` (back on the sender), at the full 77.9 Gb/s D-engine rate.

Sizing
======

An ``fft_length`` of :math:`N` spans :math:`N / 4096` packets; the receive payload buffer
must hold two windows plus the posted and slack frames (the configs use
``2 * fft_length / samples_per_packet / packets_per_frame + 32``), in huge pages. At
:math:`N = 2^{27}` and two ADCs that is about 850 huge pages plus roughly 0.6 GB of GPU
buffers per chain; one window costs about 25 ms of GPU per 41.9 ms window period.
