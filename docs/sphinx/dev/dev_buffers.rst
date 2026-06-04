.. _dev_buffers:

************
Buffers
************

More detailed info about buffers, whatever a stage writer could need to know...

Frame descriptors
=================

Buffers may carry a *frame descriptor* describing the data in each frame: a
``GenericNDArray`` (value type, extents, axis names) for ``ndarray`` buffers,
or an ``N2FrameDesc`` for ``N2`` buffers. The descriptor is declared in the
buffer's config block, built by the buffer factory at startup, and
``frame_size`` is derived from it (see :ref:`user_config`).

The model is: **buffers carry their own description; stages validate
against it.**

Rationale
---------

- Some stages work with arbitrary frame shapes, or move bytes without
  interpreting them, so a buffer's description cannot always come from the
  stages that touch it.
- Validation is the primary safety mechanism: producer, consumer, and
  config descriptions are cross-checked at startup, so mismatches appear at
  launch (with both descriptions printed) rather than downstream.
- Shapes stay visible in the config, next to the pipeline wiring, for
  review and debugging without consulting stage source.
- Descriptors are optional: ``standard`` buffers carry none, and buffers
  are typically described where shape errors would be costly.

Authoring guidance
------------------

- Write ``extents`` as expressions over the genuinely tunable config scalars
  (``samples_per_data_set``, ``num_dishes``, ...). Fixed algorithm geometry
  (tile and block sizes, packing factors) has a single correct value:
  prefer defining it in one place in the code rather than presenting it in
  config as a tunable.
- Stages with fixed shape requirements (GPU kernels and their CPU
  counterparts) should validate exactly, via
  ``Buffer::require_frame_desc(NDArray<T, D>::describe(...))`` -- a missing
  or mismatched descriptor is fatal. Stages that adapt to their input should
  check only the properties they require (value type, a particular axis,
  divisibility) and size their work from the descriptor.
- Stages that read data produced elsewhere (file readers, network receivers)
  validate the shape they discover against the buffer's declared descriptor
  with ``Buffer::require_frame_desc`` -- memory is allocated from the config
  before any data arrives. ``Buffer::set_frame_desc`` (attach on first use,
  check thereafter) remains for shapes derived in code at runtime, e.g. GPU
  pipeline copy-out.
