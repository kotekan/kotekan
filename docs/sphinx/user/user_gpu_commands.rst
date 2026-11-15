.. _gpu_commands:

************
GPU commands
************

A ``cudaProcess`` stage runs a list of ``commands`` on one GPU. Each command is
a ``cudaCommand`` (``lib/cuda``) that queues one kernel or one copy. This page
says what each command computes. The config parameters and the memory layout of
each array are in the doxygen block of the class. How a stage runs its commands
(streams, GPU memory, ring buffers) is in :doc:`/overview_theory_of_operation`.

The commands pass data to each other in GPU memory. Most arrays are ring
buffers along their time axis, so a command reads as many samples as are ready
and its output cadence need not match the frame length.

.. contents::
   :local:
   :depth: 1


Data flow
=========

Visibilities and RFI statistics (``config/chord/pathfinder.j2``):

.. code-block:: text

    E, pl_mask ---> cudaRFIS012 ---> S012 -+-> cudaRFISKtilde ---> SKtilde, RFImask
                                           +-> cudaRFIS012bar ---> S012bar
                                                 S012bar ---> cudaRFISKbar ---> SKbar, SKbartilde

    pl_mask ---> cudaPLMaskExpander ---> pl_mask_exp
    pl_mask_exp, RFImask ---> cudaPL1bitCorrelator ---> n2k_counts
    E, RFImask           ---> cudaCorrelator       ---> n2k_correlation

Beamforming:

.. code-block:: text

    E ---> cudaUpchannelizer ---> Ebar ---> cudaFRBBeamformer ---> I
    I, W2 ---> cudaFRBBeamReformer ---> I2 ---> cudaQuantize4 / cudaQuantize8
    beam positions ---> cudaCalcFRB2Weights ---> W2

    E, A, s ---> cudaBasebandBeamformer ---> J

Each beamforming output has a packet loss mask, made by a ``cudaPLMask...``
command from ``pl_mask_exp``.


Notation
========

An array is written with the name the code gives it (its ``quantity``), and an
index is the lower-case form of the axis label in the code. Sizes are the
upper-case form: there are :math:`D` dishes and :math:`F` frequencies.

.. list-table::
   :header-rows: 1
   :widths: 12 18 70

   * - Index
     - Axis label
     - Meaning
   * - :math:`t`
     - ``T``
     - Voltage time sample: one F-engine sample of one coarse channel.
   * - :math:`f`
     - ``F``
     - Coarse frequency channel on this node.
   * - :math:`p`
     - ``P``
     - Polarization (0 or 1).
   * - :math:`d`
     - ``D``
     - Dish.
   * - :math:`e`, :math:`i`, :math:`j`
     - ``P``, ``D``
     - Element: one polarization of one dish, :math:`e = (p, d)`. There are
       :math:`2D` elements. The n2k code says "station".
   * - :math:`t_\mathrm{rfi}`
     - ``Trfi``
     - Block of ``rfi_downsampling_factor`` voltage samples.
   * - :math:`\bar t_\mathrm{rfi}`
     - ``Trfibar``
     - Block of ``rfi_second_downsampling_factor`` ``Trfi`` samples.
   * - :math:`t_c`
     - ``Tc``
     - Sub-integration: block of ``sub_integration_ntime`` voltage samples.
   * - :math:`u`
     -
     - Fine channel in a coarse channel, :math:`0 \le u < U`. :math:`U` is the
       upchannelization factor.
   * - :math:`\bar f`
     - ``Fbar``
     - Upchannelized frequency, :math:`\bar f = U (f - F_\mathrm{min}) + u`.
   * - :math:`\bar t`
     - ``Tbar``
     - Upchannelized time sample: :math:`U` voltage samples.
   * - :math:`\tilde t`
     - ``Ttilde``
     - FRB time sample: :math:`T_d` upchannelized samples.
   * - :math:`m, n`
     - ``dishM``, ``dishN``
     - Position of a dish on the regular :math:`M \times N` dish grid.
   * - :math:`b_P, b_Q`
     - ``beamP``, ``beamQ``
     - FRB grid beam, :math:`0 \le b_P < 2M`, :math:`0 \le b_Q < 2N`.
   * - :math:`r`
     - ``R``
     - FRB sky beam.
   * - :math:`b`
     - ``B``
     - Baseband beam.

A sum such as :math:`\sum_{t \in t_c}` runs over the voltage samples of one
block.

The notes listed under `References`_ use :math:`\tau` for time, :math:`\pi`
for polarization, :math:`\beta` for a beam and :math:`p, q` for the FRB grid
beam, because :math:`t` and :math:`b` there mean a GPU thread and a byte.

Some names have two meanings, and this page keeps both because the code does:

* For the RFI arrays, ``bar`` means downsampled in time and ``tilde`` means
  summed or averaged over feeds (``S012bar``, ``SKtilde``). For the
  beamforming arrays, ``bar`` means upchannelized and ``tilde`` means
  downsampled to the FRB cadence (``Ebar``, ``Tbar``, ``Ttilde``).
* :math:`M` is the number of taps in the upchannelizer, and :math:`(M, N)` is
  the dish grid in the FRB beamformer.
* :math:`W` is the window of the upchannelizer, ``W`` the weights of the FRB
  beamformer and ``W2`` the weights of the FRB beam reformer.
* ``S`` labels the axis that holds :math:`(S_0, S_1, S_2)`, and is also the
  dish layout array of the FRB beamformer.

Voltages
--------

``E[t, f, p, d]`` is the complex voltage with 4-bit real and imaginary parts
(``int4x2_swapped_withoffset``). The parts are in :math:`[-7, 7]`. A lost
sample is zero.

Masks
-----

In every mask, 1 means good.

* ``pl_mask`` is the packet loss mask. It has one bit for every 2 time
  samples, 4 frequencies and 8 dishes of one polarization.
* ``pl_mask_exp`` is the same mask with one bit for every time sample and
  frequency.
* ``bf_mask[p, d]`` is the bad feed mask.
* ``RFImask[f, t]`` has one bit for every voltage sample and frequency.

Masks are packed in bits. A pair of labels such as ``T2hi64`` and ``T2lo64``
is one packed time axis: ``hi`` counts 64-bit words and ``lo`` counts the
bytes in a word. The ``2`` says that one bit covers two time samples. ``F4``
and ``D8`` are frequency and dish axes downsampled by 4 and 8.

Time axes
---------

.. list-table::
   :header-rows: 1
   :widths: 14 46 20 20

   * - Axis
     - Voltage samples for each element
     - Pathfinder
     - Duration
   * - ``T``
     - 1
     - 1
     - 5.12 µs
   * - ``Trfi``
     - ``rfi_downsampling_factor``
     - 256
     - 1.31 ms
   * - ``Trfibar``
     - ``rfi_downsampling_factor * rfi_second_downsampling_factor``
     - 8192
     - 41.9 ms
   * - ``Tc``
     - ``sub_integration_ntime``
     - 8192
     - 41.9 ms
   * - ``Tbar``
     - :math:`U`
     -
     -
   * - ``Ttilde``
     - :math:`U T_d`
     - 192
     - 0.98 ms


Copies and synchronization
==========================

These commands compute nothing.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Purpose
   * - ``cudaInputData``
     - Copies a host frame to GPU memory.
   * - ``cudaOutputData``
     - Copies GPU memory to a host frame, with its metadata.
   * - ``cudaOutputDataZero``
     - Copies zeros to GPU memory, to initialize an output.
   * - ``cudaCopyToRingbuffer``
     - Copies host frames into a GPU ring buffer.
   * - ``cudaCopyNToRingbuffer``
     - Merges frames from several host buffers into one GPU ring buffer
       (CHIME).
   * - ``cudaCopyFromRingbuffer``
     - Copies from a GPU ring buffer to host frames.
   * - ``cudaSyncInput``, ``cudaSyncOutput``, ``cudaSyncStream``
     - Make one CUDA stream wait for another: the kernels for the input
       copy, or the output copy for the kernels.


Packet loss masks
=================

``cudaPLMaskExpander``
    Writes ``pl_mask_exp`` from ``pl_mask``. Each input bit is copied to the
    2 time samples and 4 frequencies that it covers.

``cudaPL1bitCorrelator``
    Counts the samples that go into each visibility. With :math:`a` and
    :math:`b` each a group of 8 dishes of one polarization (``D8``):

    .. math::

        \mathrm{counts}[t_c, f, a, b] =
            \sum_{t \in t_c} \mathrm{RFImask}[f, t] \;
            \mathrm{pl}[t, f, a] \; \mathrm{pl}[t, f, b]

    The matrix is stored as the 8 x 8 tiles of its lower triangle.

``cudaPLMaskAccumulator``
    Counts the good samples of each input in each sub-integration:
    :math:`\mathrm{pl\_counts}[t_c, f, p, d] = \sum_{t \in t_c} \mathrm{pl}[t, f, p, d]`.

``cudaPLMaskUpchannelizer``
    The mask of ``Ebar``. An upchannelized sample depends on :math:`MU`
    voltage samples, so it is good only if all of them are good.

``cudaPLMaskFRB1``
    The mask of ``I``, one byte for each :math:`(\tilde t, \bar f)`. An FRB
    sample is good only if all :math:`T_d` input samples are good for all
    dishes and polarizations.

``cudaPLMaskBasebandBeamformer``
    The mask of ``J``. A beam sample is good only if all dishes are good at
    its time, frequency and polarization. The mask is the same for all beams.

:ref:`packet_loss` follows a lost packet through the CHIME pipeline.


RFI statistics
==============

The spectral kurtosis (SK) estimator tests if the voltages of one frequency
are Gaussian noise. It needs three sums for each element.

``cudaRFIS012``
    Writes ``S012[t_rfi, f, S, p, d]``, where the ``S`` axis holds

    .. math::

        S_0 = \sum_{t \in t_\mathrm{rfi}} \mathrm{pl}[t], \qquad
        S_1 = \sum_{t \in t_\mathrm{rfi}} |E[t]|^2, \qquad
        S_2 = \sum_{t \in t_\mathrm{rfi}} |E[t]|^4

    :math:`S_0` is the number of good samples. A lost sample adds nothing to
    :math:`S_1` and :math:`S_2`, because its voltage is zero.

``cudaRFIS012bar``
    Sums ``S012`` over ``rfi_second_downsampling_factor`` time samples and
    writes ``S012bar[tbar_rfi, f, S, p, d]``.

``cudaRFIS012tilde``
    Sums ``S012`` over the good feeds of ``bf_mask`` and writes
    ``S012tilde[t_rfi, f, S]``.

``cudaRFISKtilde`` and ``cudaRFISKbar``
    Both run the n2k SK kernel. For each element, with
    :math:`\mu = S_1 / S_0` the mean power:

    .. math::

        \mathrm{SK} = \frac{S_0 + 1}{S_0 - 1}
            \left( \frac{S_0 S_2}{S_1^2} - 1 \right) - b, \qquad
        \sigma = \frac{\sigma_0(\mu)}{\sqrt{S_0}}

    SK is 1 on average for Gaussian noise, and :math:`\sigma` is its
    expected scatter. The bias :math:`b(\mu, 1/S_0)` and
    :math:`\sigma_0(\mu)` correct for the 4-bit quantization. They come from
    a table that the kernel interpolates in :math:`\log \mu`.

    An element is valid if :math:`S_0` is at least
    ``rfi_single_feed_min_good_frac`` of the block and
    ``rfi_mu_min`` :math:`\le \mu \le` ``rfi_mu_max``.

    The feed average weights each valid, good feed by :math:`w_e = S_0`:

    .. math::

        \widetilde{\mathrm{SK}} = \frac{\sum_e w_e \mathrm{SK}_e}{\sum_e w_e},
        \qquad
        \tilde b = \frac{\sum_e w_e b_e}{\sum_e w_e},
        \qquad
        \tilde\sigma = \frac{\sqrt{\sum_e w_e^2 \sigma_e^2}}{\sum_e w_e}

    It is valid if :math:`\sum_e w_e` is at least
    ``rfi_feed_averaged_min_good_frac`` of all samples of all elements.

    The outputs have an ``SK`` axis that holds :math:`(\mathrm{SK}, b, \sigma)`.
    An invalid entry is :math:`(0, 0, -1)`, so test for :math:`\sigma < 0`.

    .. list-table::
       :header-rows: 1
       :widths: 22 18 60

       * - Command
         - Input
         - Output
       * - ``cudaRFISKtilde``
         - ``S012``
         - ``SKtilde[t_rfi, f, SK]``, the feed average, and ``RFImask``.
       * - ``cudaRFISKbar``
         - ``S012bar``
         - ``SKbar[tbar_rfi, f, SK, p, d]`` for each element, and its feed
           average ``SKbartilde[tbar_rfi, f, SK]``.

    ``RFImask[f, t]`` is good for all voltage samples of a block if the feed
    average is valid and

    .. math::

        |\widetilde{\mathrm{SK}} - 1| \le
            \mathtt{rfi\_sk\_rfimask\_sigmas} \times \tilde\sigma

    The correlator uses ``RFImask`` (first-stage excision). The CPU stage
    ``RfiFrameMask`` uses ``SKtilde`` to reject whole sub-integrations
    (second-stage excision).

``cudaPilotProxyDetector``
    Detects DTV pilot tones in the voltages. See :ref:`pilotproxy`. Its mask
    ``dtv_mask[f]`` is the exception to the rule for masks: 1 means reject.
    The CPU stage ``DtvRfiMask`` clears the ``RFImask`` bits of a rejected
    frequency. In ``pathfinder.j2`` the two correlators read this merged
    mask.


Visibilities
============

``cudaCorrelator``
    Runs the n2k correlator. For each sub-integration and frequency, with
    :math:`i` and :math:`j` elements:

    .. math::

        V[t_c, f, i, j] = \sum_{t \in t_c} \mathrm{RFImask}[f, t] \;
            E[t, f, i] \; E[t, f, j]^*

    The output ``n2k_correlation`` holds the real and imaginary parts as
    ``int32``, as the 16 x 16 tiles of the lower triangle. The sum is not
    normalized: ``N2Accumulate`` divides it by ``n2k_counts`` from
    ``cudaPL1bitCorrelator``, and writes the upper triangle.

    ``sub_integration_ntime`` must be a multiple of 256.


Upchannelizer
=============

``cudaUpchannelizer_<setup>_U<U>_K<K>``
    Splits each coarse channel into :math:`U` fine channels with a polyphase
    filter bank of :math:`M = 4` taps. The time resolution becomes :math:`U`
    times coarser. Each coarse channel, polarization and dish is independent:

    .. math::

        \mathrm{Ebar}[\bar t, \bar f, p, d] = \mathrm{Quantize}_K \left(
            G[\bar f] \sum_{s=0}^{MU-1} W(s) \,
            e^{-2\pi i \, (u - (U-1)/2) \, s / U} \,
            E[U \bar t + s, f, p, d] \right)

    with the sinc-Hanning window

    .. math::

        W(s) = \frac{1}{U} \cos^2(\pi s') \,
            \frac{\sin(\pi M s')}{\pi M s'}, \qquad
        s' = \frac{s - (MU - 1)/2}{MU}

    The half-integer :math:`(U-1)/2` in the phase puts the :math:`U` fine
    channels symmetrically about the centre of the coarse channel.
    ``G[fbar]`` is a real gain. :math:`\mathrm{Quantize}_K` rounds and clamps
    the real and imaginary parts to :math:`K` bits (4 or 8).

    Only coarse channels ``Fmin`` to ``Fmax - 1`` are processed. Kernels with
    different :math:`U` skip different numbers of leading samples
    (``max_upchannelization_factor``), so that their outputs are centred on
    the same times.

    ``lib/cuda/upchannelizeReference.hpp`` is a serial version of this
    equation. ``cudaUpchannelize`` and ``cudaUpchannelize16k`` are earlier
    kernels with fixed sizes.


FRB beamformer
==============

The FRB beams are made in two steps. The first forms a regular grid of beams
with an FFT. The second interpolates the grid to the wanted sky positions.

``cudaFRBBeamformer_<setup>_U<U>``
    Puts the dishes on the regular :math:`M \times N` dish grid
    (``S[d]`` gives the position :math:`(m, n)` of dish :math:`d`; an empty
    position is zero), applies the complex weights ``W``, and does a 2-d FFT
    that is zero-padded by 2:

    .. math::

        \tilde E[\bar t, \bar f, p, b_P, b_Q] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1}
            W[\bar f, p, m, n] \; \mathrm{Ebar}[\bar t, \bar f, p, m, n] \;
            \exp \left( \frac{2\pi i \, m \, b_P}{2M}
                      + \frac{2\pi i \, n \, b_Q}{2N} \right)

    It writes the power, summed over polarization and :math:`T_d` time
    samples:

    .. math::

        I[\tilde t, \bar f, b_P, b_Q] = \sum_{\bar t \in \tilde t} \; \sum_p
            \left| \tilde E[\bar t, \bar f, p, b_P, b_Q] \right|^2

    The grid beam :math:`(b_P, b_Q)` points at the sky position
    :math:`(\theta, \theta') = (b_P/2, b_Q/2)`, where

    .. math::

        \theta = \frac{M \, (\hat n \cdot \vec\sigma_M)}{\lambda}, \qquad
        \theta' = \frac{N \, (\hat n \cdot \vec\sigma_N)}{\lambda}

    for a unit vector :math:`\hat n` to the sky and the dish spacings
    :math:`\vec\sigma_M`, :math:`\vec\sigma_N` along the two grid axes.

    There is one kernel for each :math:`U`. All of them write to one ``I``
    array, each to its own range of ``Fbar``, and :math:`U T_d` is the same
    for all, so ``I`` has one time axis.

``cudaCalcFRB2Weights``
    Writes the weights that interpolate the grid to sky beam :math:`r` at
    :math:`(\theta_r, \theta'_r)`:

    .. math::

        W2[\bar f, r, b_Q, b_P] = U^M_{b_P}(\theta_r) \; U^N_{b_Q}(\theta'_r),
        \qquad
        U^M_p(\theta) = \frac{1}{M} \sum_{s=0}^{M} A_s
            \cos \left( \frac{\pi (2\theta - p) s}{M} \right)

    with :math:`A_0 = A_M = 1/2` and :math:`A_s = 1` otherwise.
    :math:`\theta` depends on frequency, so there is one matrix for each
    ``Fbar``.

``cudaFRBBeamReformer``
    Interpolates the grid beams to the sky beams. This is exact, not an
    approximation: the grid beams are a complete basis for the power.

    .. math::

        I2[\tilde t, \bar f, r] = \sum_{b_P, b_Q}
            W2[\bar f, r, b_Q, b_P] \; I[\tilde t, \bar f, b_P, b_Q]

    The sum is one cuBLAS matrix product for each frequency.

``cudaHFB1Accumulate``
    Averages ``I`` over ``hfb_second_downsampling_factor`` time samples.

``cudaQuantize4`` and ``cudaQuantize8``
    Quantize ``I2`` in chunks. For each chunk, the mean and standard
    deviation of the good samples give an ``offset`` (the mean) and a
    ``scale``, so that ``stddev_cutoff`` standard deviations fill the integer
    range. A value is recovered as :math:`\mathrm{offset} + \mathrm{scale}
    \times n`. A NaN or a lost sample is :math:`n = -8` (4 bits) or
    :math:`n = 0` (8 bits).

``cudaCHIMEFRBBeamformer_<setup>``, ``cudaTranspose2048_chime``, ``cudaQuantize8Chime``
    The CHIME forms of the FRB beamformer, of the input reordering before
    it, and of the quantizer.


Baseband beamformer
===================

``cudaBasebandBeamformer_<setup>``
    Forms :math:`B` voltage beams. Each frequency and polarization is
    independent. ``A`` is a complex phase with 8-bit real and imaginary
    parts, and ``s`` is a bit shift that acts as a gain:

    .. math::

        J[t, f, p, b] = \mathrm{Quantize}_4 \left(
            \Big( \sum_d A[f, p, b, d] \; E[t, f, p, d] \Big)
            \gg s[f, p, b] \right)

    Here :math:`x \gg s` is a division by :math:`2^s` that rounds to the
    nearest integer, and :math:`\mathrm{Quantize}_4` clamps the real and
    imaginary parts to :math:`[-7, 7]`. All arithmetic is in integers.

    The kernel does the sum in two parts. It sums groups of dishes and shifts
    by :math:`\sigma` bits, then sums the groups and shifts by
    :math:`s - \sigma` bits. :math:`\sigma` is fixed in each kernel (2 for the
    pathfinder, 3 for CHORD), and :math:`s` must be larger. The two rounding
    steps matter only for a test that compares bit for bit.

    ``A`` changes slowly: there is one matrix for each
    ``bb_phase_lifetime_in_samples`` voltage samples.


References
==========

The kernels follow notes by Kendrick Smith. Their notation differs from this
page as described under `Notation`_.

* *CHORD GPU upchannelization*, ``julia/docs/CHORD_GPU_upchannelization.pdf``.
  Eq. (83) is the kernel.
* *CHORD FRB beamformer*, ``julia/docs/CHORD_FRB_beamformer.pdf``. Eqs. (50)
  and (51) are the first kernel. Eqs. (6), (7) and (137) are the weights and
  the second kernel.
* *CHORD 8-bit baseband beamformer*,
  ``julia/docs/CHORD_8_bit_baseband_beamformer.pdf``. Eq. (1) is the kernel,
  and Appendix A is the exact integer arithmetic.
* *Note on CHIME (not CHORD) FRB beamforming*,
  ``julia/docs/CHIME_FRB_beamforming.pdf``.
* *Notes on the SK-statistic*: the bias and variance of SK for quantized
  voltages.
* The n2k headers in ``external/n2k/include/n2k`` describe the correlator, the
  packet loss kernels and the RFI kernels.
