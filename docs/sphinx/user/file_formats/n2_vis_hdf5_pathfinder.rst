.. _n2_vis_pathfinder:

=====================================
CHORD pathfinder visibility files
=====================================

This page describes the :ref:`hdf5N2Write <hdf5N2Write>` output as
configured for the CHORD pathfinder receiver
(``config/chord_pathfinder_recv.j2``). It supplements the generic format
reference, :ref:`n2_vis_file_format`, with this deployment's array sizes,
binning, compression settings, and pipeline-specific caveats.

The two writer instances
========================

Two writer instances run in parallel, each writing acquisitions under its
own ``base_dir``: a *full* tree (all correlator inputs, no eigen-data) and
a *subset* tree (an 8-input subset with 2 eigenvalues/vectors).

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * -
     - ``hdf5_N2_write_full``
     - ``hdf5_N2_write_subset``
   * - Inputs (``num_elements``, :math:`N_e`)
     - 128
     - 8
   * - Products (``num_prod``, :math:`N_p`)
     - 8256
     - 36
   * - Eigenpairs (``num_ev``, :math:`N_{ev}`)
     - 0
     - 2
   * - Chunk cap, frequency (``blocksize_f``)
     - 32
     - 16
   * - ``/vis`` chunks
     - (32, 16, 20)
     - (16, 16, 20)
   * - ``/vis`` size (uncompressed)
     - ~8.1 GB
     - ~35 MB

Both instances share all other parameters: :math:`N_t` = ``num_file_t`` =
20 time bins per file, visibility layout ``FullUpperTri``, element ordering
``CHORDEarly``, bitshuffle + LZ4 compression
(``use_bitshuffle: true``, ``compression: "lz4"``), a 30 s
``late_frame_grace_seconds``, and digital gains fetched over HTTP from the
FPGA controller (the ``/fpga_controller`` config block, via
``baseband_gain_host_info``) and cached as
``<acq>/.partial/baseband_gains.h5``.

The *subset* stream contains the cross-products of correlator inputs 0--7
only (the six connected feeds ``A01p1``, ``A01p2``, ``A06p1``, ``A06p2``,
``A07p1``, ``A07p2`` plus two unconnected inputs; see ``/index_map/label``).
The *full* stream contains all :math:`128 \times 129 / 2 = 8256` products of
the 128 inputs.

Frequencies and time bins
=========================

**Frequency axis.** The science band is 300--1500 MHz in channels of width
:math:`3200/16384 = 0.1953125` MHz, giving :math:`N_f = 6145` channels
(``freq_id`` 1536--7680 of the telescope's 8192). ``/index_map/freq`` runs
from centre 300.0 to 1500.0 MHz.

**Time bins.** The upstream accumulator runs in ERA-binning mode
(``bin_in_ERA: true``, ``num_bins_per_rotation: 8640``): bins are
:math:`1/24^\circ` of Earth Rotation Angle each, so ``bin_start_ERA_deg``
and ``bin_end_ERA_deg`` are exact multiples of :math:`1/24^\circ`. With the
configured accumulation of 238 sub-integrations of 8192 FPGA ticks, the
nominal accumulated length per bin (``frame_length_fpga_ticks``) is
:math:`1\,949\,696` ticks :math:`\times\ 5120\,\mathrm{ns} \approx 9.982` s.
Each file groups 20 consecutive bins and spans approximately 200 s.

**Key attribute values** following from this configuration:
``fpga_seq_length_ns`` = 5120, ``num_telescope_f`` = 8192,
``num_dishes`` = 64, ``nyquist_zone`` = 1, and ``dish_coelev_deg`` = -27.3
(approximately a Taurus A transit pointing).

Pipeline caveats
================

In this pipeline no gain-application, radiometer-test, or input-flagging
stage runs, so the following hold sentinels throughout (see the *Sentinels*
convention on the generic page):

* ``gain``: -1+0j (the applied F-engine gains are in ``/digital_gains``);
* ``flags``: 0;
* ``radiometer_chi2``: -1;
* ``rfi_frame_excision_enabled``: false, with ``rfi_frame_excision_num`` 0.

The *subset* stream carries real eigen-data: ``EigenN2Iter`` stages run
upstream on it, so ``eval``, ``evec``, and ``erms`` are populated. The
*full* stream is written with ``num_ev`` = 0 (its eigen axes have zero
extent).

Acquisitions may end with partially filled files (end of run, or sender
restarts), so always apply the ``/frames_added`` mask. Note also that
acquisitions taken while the upstream RFI excision flagged all samples are
*fully excised*: every data value (including the eigen-data) is zero and
``frac_lost`` stays at 1.0 even though ``/frames_added`` is 1, since frames
were received but contained no accumulated samples. Check
``frac_lost`` < 1 (or ``valid_fpga_count`` > 0) as well as
``/frames_added`` before interpreting values.

The ``/digital_gains`` group is a verbatim copy of the gains archive served
by the FPGA controller; its ``update_time`` axis length equals the number
of gain updates in that archive, with ``gain_coeff`` spanning the full FPGA
band (8192 channels) and all 128 inputs.

``/config_json`` holds one JSON snapshot per tracked configuration --- the
receiver process, the upstream sender processes, and the FPGA configuration
fetched from the controller.

Reading example
===============

.. code-block:: python

    import h5py, hdf5plugin  # registers the bitshuffle filter for /vis etc.
    import numpy as np

    path = "vis_<abs_idx>_<YYYYMMDDT_HHMMSS>_<ns>.h5"  # a subset-stream file

    with h5py.File(path) as f:
        freq = f["index_map/freq"]["centre"]      # MHz, (6145,)
        prod = f["index_map/prod"][:]             # (input_a, input_b), (36,)
        good = f["frames_added"][:].astype(bool)  # (6145, 20) validity mask
        vis  = f["vis"][:]                        # (6145, 36, 20) complex64
        w    = f["vis_weight"][:]
        t    = f["time_center_t_inst_ns"][:] * 1e-9   # UNIX seconds
        vis[~good[:, None, :].repeat(vis.shape[1], 1)] = np.nan
