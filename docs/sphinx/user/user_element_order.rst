.. _element_order:

****************
Element Ordering
****************

The *element* axis of kotekan data arrays (visibilities, gains, flags,
baseband, ...) indexes the telescope's correlator inputs. Different
telescopes, and different parts of the pipeline for the same telescope,
arrange the same physical inputs in memory in different orders, so an
element index is only meaningful together with an *element ordering*. The
``ElementOrder`` enum (``lib/utils/Telescope.hpp``) names the defined
conventions, and the ``Telescope`` class converts between element indices,
canonical input identities, and physical geometry.

Defined orderings
=================

Configuration options and file attributes use the enum names as strings.

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Name
     - Arrangement
     - Description
   * - ``CHIMECorrelator``
     - table-defined
     - Order as received by the CHIME X-engine; slightly scrambled, defined
       via the input-reorder table.
   * - ``CHIMECylinder``
     - ``[C][P][CF]``
     - Cylinders west to east, then polarization, then feeds within a
       cylinder south to north.
   * - ``CHIMEBeamformer``
     - ``[P][D]``
     - Polarization slowest, then feeds ordered as in ``CHIMECylinder``.
   * - ``CHORDEarly``
     - ``[D][P]``
     - Pre-pathfinder CHORD: dish-major with polarization fastest ---
       element :math:`2d` is polarization 1 and element :math:`2d+1` is
       polarization 2 of dish :math:`d`. Dish ordering is defined by the
       telescope's ``dish_inputs`` table.
   * - ``CHORDBeamformer``
     - ``[P][D]``
     - Production CHORD: polarization slowest --- element
       :math:`p N_d + d` is polarization :math:`p` of dish :math:`d`. Dish
       ordering is defined by the table.

Station IDs and geometry
========================

Internally each physical input has an ordering-independent *station ID*
(``station_id_t``). Conversions always go through it:

* ``Telescope::element_index_to_station_id(el_idx, order)`` and the inverse
  ``station_id_to_element_index(st_id, order)`` translate between an
  element index in a given ordering and the canonical input. For CHORD the
  station ID encodes (dish, polarization) as
  :math:`d + p N_d`; for CHIME it encodes (cylinder, polarization,
  feed-in-cylinder).
* ``station_id_to_main_array_grid_indices(st_id)`` returns the input's
  integer (x, y) position in the main-array grid --- x counting east, y
  counting north, from the southwest corner --- or (-1, -1) for inputs
  that are not part of the main array (RFI antennas, unpopulated inputs,
  ...).
* ``station_id_to_feed_position_m(st_id)`` returns the input's 3D position
  in the grid frame, in metres, including any per-feed displacement from
  the fiducial station position.

The convenience methods ``get_main_array_grid_indices(num_elements,
order)`` and ``get_feed_positions_m(num_elements, order)`` return these for
a whole element axis at once, in the requested ordering.

In configuration
================

Stages that interpret (rather than just relay) element-ordered data take an
``input_order`` option, set to one of the enum names above. These currently
include ``N2Accumulate``, ``N2TimeDownsample``, ``calcBBPhase``,
``hdf5N2Write``, ``hdf5FileWrite``, and the CUDA FRB beamformer kernels.
The CHORD pathfinder pipeline uses ``CHORDEarly`` throughout.

In data files
=============

The HDF5 writers record the ordering so files are self-describing:

* :ref:`n2_vis_file_format` and :ref:`hdf5_frames_format` files carry an
  ``input_order`` root/dataset attribute, plus per-element
  ``main_array_grid_indices`` (int64, :math:`N_e \times 2`) and
  ``feed_positions_m`` (float64, :math:`N_e \times 3`) attributes computed
  in that same ordering.
* All element-indexed quantities in those files, i.e. the ``element`` axes of
  datasets, and the ``input_a`` / ``input_b`` indices in
  ``/index_map/prod``, follow the recorded ``input_order``.
