.. _packet_loss:

*********************
Packet Loss in CHIME
*********************

F-engine packets can be lost on the network, or rejected on arrival. Kotekan
does not interpolate across lost samples. It sets them to zero, and keeps track
of which output samples depend on them. This page describes how a lost packet
propagates through the CHIME CUDA pipeline, and in particular which FRB output
samples it invalidates. For comparison, the last section describes how the AMD
(HSA) CHIME FRB pipeline handled packet loss.

This page describes CHIME only. It does not describe CHORD or the other
instruments, whose packet loss handling differs.

Detecting lost packets
======================

The DPDK handler ``iceBoardShuffle`` (``lib/dpdk/iceBoardShuffle.hpp``) writes a
``lost_samples`` buffer with one byte per time sample, set to 1 if the sample is
lost. A gap in the FPGA sequence numbers marks the missing samples as lost.
Packets rejected for any other reason (shuffle flags in the FPGA footer, wrong
order, bad checksum) do not advance the expected sequence number, so they end up
counted as lost as well.

Each of the four links carries 512 of the 2048 CHIME inputs for all four
frequencies of a ``lost_samples`` buffer. A packet lost on one link thus flags
its time samples for all inputs and all four frequencies.

The ``zeroSamples`` stage (``lib/dpdk/zeroSamples.cpp``) overwrites every
flagged sample of the voltage buffer with ``0x88``, which is :math:`0 + 0i` in
the offset-encoded 4+4-bit format, and adds the number of lost samples to the
frame's metadata. Everything downstream therefore sees zeros, not stale data.

The packet loss mask on the GPU
===============================

``lostSamplesToPLMask`` packs the ``lost_samples`` buffers into a one-bit packet
loss (PL) mask ``pl_mask``. A bit is 1 if the sample is valid and 0 if it is
lost. Each bit covers two consecutive samples (a pair is lost if either sample
is). The mask has bits for each group of four frequencies, polarization and
group of eight dishes; since a lost CHIME packet affects all inputs,
``lostSamplesToPLMask`` sets them alike for all dishes and polarizations. On the
GPU, ``cudaPLMaskExpander`` expands it to the expanded mask
``pl_mask_exp`` with one bit per sample, frequency, polarization and group of
eight dishes. The RFI spectral-kurtosis kernels and the N2 lost-sample counts
read the PL mask, so that lost samples do not enter their statistics.

Each upchannelizer has a ``cudaPLMaskUpchannelizer`` that carries the mask onto
its time grid. An upchannelized sample is the output of an :math:`M`-tap PFB
(:math:`M = 4`) with upchannelization factor :math:`U`. It depends on
:math:`M U` consecutive input samples, and is valid only if all of them are. The
PL mask upchannelizer skips the same samples at startup as the voltage
upchannelizer, so that both outputs begin at input sample
:math:`(M-1) U_\text{max} / 2` and carry the same ``fpga_seq_num``.

The FRB output mask
===================

An FRB output sample is invalid if it depends on any lost input sample. The FRB1
and FRB2 beams themselves are not changed; instead the pipeline produces a mask
of the FRB output, with one byte per output sample and frequency: 1 if the
sample is valid, 0 if not. It has the dimensions ``[Ttilde][Fbar]``, the same
time samples and frequencies as the FRB1 beams. The quantizer reads it and
encodes invalid samples as 0 (see below), so the mask is not needed after
quantization; it is copied to the host (``host_frb_pl_mask_buffer``) and written
only for debugging (``do_once_and_dump``).

``cudaPLMaskFRB1`` calculates this mask. It is the PL mask counterpart of the
FRB1 beamformer: an FRB1 output sample sums :math:`T_d` consecutive
upchannelized samples over all dishes and both polarizations, so it is valid
only if the PL mask is valid for all of them. CHIME upchannelizes all FRB
frequencies by :math:`U = 16` and downsamples by :math:`T_d = 24`, so its input
is the PL mask upchannelized by 16.

Altogether, an FRB1 output sample depends on :math:`(T_d + M - 1) U` input
samples: it covers 384 input samples and depends on 432. With the 4-tap PFB,
these overlap the input samples of the neighbouring output samples by
:math:`(M-1) U / 2 = 24` on each side. A lost packet near the boundary of an
FRB1 output sample therefore also invalidates the neighbouring sample. Because the PL mask flags samples in pairs, a window that begins or ends
in the middle of a pair may include one more lost sample than necessary; this
only invalidates more samples, never fewer.

The FRB2 beamformer forms each output beam from the FRB1 beams at the same time
and frequency, so the mask also applies to the FRB2 beams.

The quantizer
=============

``cudaQuantize8`` quantizes the FRB2 beams to 8 bits in chunks of 16 times and
16 frequencies, each with its own offset and scale. It reads the FRB output mask
and skips invalid samples when choosing a chunk's offset and scale: the minimum
and maximum of the valid samples map to the codes 1 and 254. Invalid samples are
encoded as 0, the code that also represents not-a-number, as in the AMD
pipeline. A chunk without valid samples has offset and scale 0. The quantizer checks that the mask describes the same samples and
frequencies as the beams.

The AMD CHIME FRB pipeline
==========================

The production CHIME FRB pipeline ran on AMD GPUs, using the HSA code that was
removed from kotekan in commit ``42d3af4ef`` (May 2025). Its configuration is
``config/chime_science_run_gpu.yaml``, and the GPU files below refer to the
parent of that commit; the CPU stage ``frbPostProcess`` is still in
``lib/stages``. It detected and zeroed lost packets exactly as described above, but
did not use a PL mask on the GPU:

- ``hsaRfiZeroData`` zeroed the voltages of RFI-flagged samples in place, but
  ran after ``hsaBeamformReorder`` had copied them for the beamformer. RFI
  zeroing thus affected only the N2 products, not the FRB beams.
- The beamformer (``frb_beamform_amd.cl``) and the upchannelizer
  (``frb_upchan_amd.cl``) had no mask. The upchannelizer used a plain 128-point
  FFT without PFB. It summed 3 FFTs and both polarizations, and reduced 128 to
  16 fine frequencies. An output sample thus covered exactly 384 input samples,
  aligned with the frame, and losses did not reach the neighbouring samples.
- ``frbPostProcess`` combined the outputs of the four GPUs into packets. It
  marked an output sample as dropped if any of its 384 input samples was lost,
  for all four frequencies. It set dropped samples to zero in the incoherent
  beams, encoded them as byte 0 (0 and 255 are reserved; data use 1 to 254),
  and skipped them when choosing each packet's scale and offset. A packet whose
  samples were all dropped had scale and offset 0 and was counted in
  ``kotekan_frb_masked_packets_total``.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Stage
     - AMD (HSA)
     - CUDA
   * - Lost samples
     - flagged by ``iceBoardShuffle``, zeroed by ``zeroSamples``
     - the same
   * - Mask on the GPU
     - none
     - ``pl_mask``, used by RFI, N2 and the FRB output mask
   * - Upchannelizer
     - 128-point FFT, no overlap between output samples
     - 4-tap PFB; a loss can also invalidate a neighbouring FRB sample
   * - FRB output
     - dropped samples encoded as byte 0 in the FRB packets
     - invalid samples encoded as byte 0
   * - Quantizer
     - skips dropped samples for scale and offset
     - skips invalid samples for scale and offset
