.. _dev_buffers:

************
Buffers
************

Buffers connect stages: a stage (optionally) reads from input buffers and
(optionally) writes to output buffers, and the buffer takes care of the
synchronization between them, so stages never signal each other directly.

The core ``Buffer`` is a fixed-size, supporting multi-producer multi-consumer
access to a pool of frames. Each frame is a contiguous block of memory, with
``num_frames`` frames of ``frame_size`` bytes each. A producer asks for an
*empty* frame (``wait_for_empty_frame``), fills it, and marks it *full*
(``mark_frame_full``); a consumer waits for a *full* frame
(``wait_for_full_frame``), reads it, and marks it *empty*
(``mark_frame_empty``). A frame is recycled once every registered consumer
has marked it empty. All of these calls are thread safe and, used
correctly, deadlock free.

Points a stage writer should know:

- Producers and consumers each register with the buffer by name; a buffer
  may have several of each. Consumers must never write to frames. A buffer
  with no registered consumer drops its frames (logging an INFO).
- Each frame can carry a metadata object drawn from the buffer's
  ``metadata_pool``, and -- depending on the buffer type -- the buffer
  carries a frame descriptor (see below).
- Frame memory is not zeroed between uses: a producer must overwrite (or
  zero) the whole frame unless zeroing is explicitly enabled.
- Buffers are declared in the YAML config as ``kotekan_buffer:`` blocks
  and built by the buffer factory at startup. The buffer's name is the
  path of its config block, and stages receive their buffers through the
  ``bufferContainer``.
- ``RingBuffer`` is the exception to the frame model: it coordinates
  access to a ring of elements in memory it does not itself own (e.g. a
  ring in GPU memory), and its producers and consumers may run at
  different cadences.

Buffer types
============

``kotekan_buffer`` selects the buffer type. Every buffer block takes
``log_level`` (usually inherited) and optionally ``metadata_pool`` and
``numa_node``. Frame-oriented buffers (all types except ``ring``) also
require ``num_frames``, plus the optional allocation tunables
``use_hugepages`` (off), ``mlock_frames`` (on), ``zero_new_frames`` (on),
``zero_value`` (0), and ``cpu_affinity`` (unset).

Type-specific parameters are required unless marked *optional*:

.. list-table::
   :header-rows: 1
   :widths: 12 48 40

   * - Type
     - Type-specific parameters
     - Frame descriptor
   * - ``ndarray``
     - *Structural (required):*

       - ``value_type`` -- kotekan ``DataType`` name (``float32``,
         ``int4x2``, ...)
       - ``extents`` -- list of dimension sizes; entries may be arithmetic
         expressions over other config values

       *Labels (optional):*

       - ``quantity_name`` -- label for the stored quantity
       - ``dimnames`` -- list of axis labels, one per extent

       Omitted labels are supplied by the producing stage (see below).
       ``frame_size`` is derived from the descriptor.
     - ``GenericNDArray``, built from the config block and attached at
       startup
   * - ``N2``
     - - ``num_elements`` -- number of correlator inputs
       - ``num_ev`` -- number of eigenvectors
       - ``n2_layout`` -- product layout (``FullUpperTri``,
         ``Autocorrelations``, ...; see ``N2Layout``)

       ``frame_size`` is derived from the descriptor.
     - ``N2FrameDesc``, built from the config block and attached at
       startup
   * - ``standard``
     - - ``frame_size`` -- frame size in bytes, set explicitly
     - none declared; stages may attach one at runtime via
       ``ensure_frame_desc`` / ``require_frame_desc``
   * - ``vis``
     - - ``num_elements`` -- number of correlator inputs
       - ``num_ev`` -- number of eigenvectors
       - ``num_prod`` (*optional*) -- number of products; defaults to the
         full triangle

       ``frame_size`` is computed from these.
     - none (layout is handled by ``VisFrameView``)
   * - ``hfb``
     - - ``num_frb_total_beams`` -- number of FRB beams
       - ``factor_upchan`` -- upchannelization factor (sub-frequencies)

       ``frame_size`` is computed from these.
     - none (layout is handled by ``HFBFrameView``)
   * - ``ring``
     - - ``ring_buffer_size`` -- ring capacity in bytes (the API speaks of
         abstract *elements*, but all current users count bytes); the
         buffer manages access only, the memory is owned elsewhere
     - none

For new pipelines, prefer ``ndarray`` (or ``N2`` for correlation products)
wherever a frame is a typed array: the descriptor documents the shape in
the config and lets stages validate it at startup. ``standard`` remains
for opaque or byte-oriented frames.

Frame descriptors
=================

Buffers may carry a *frame descriptor* describing the data in each frame: a
``GenericNDArray`` for ``ndarray`` buffers,
or an ``N2FrameDesc`` for ``N2`` buffers. The descriptor is declared in the
buffer's config block, built by the buffer factory at startup, and
``frame_size`` is derived from it (see :ref:`user_config`).

For an ``ndarray`` descriptor the **structural** fields (``value_type``,
``extents``) are required -- they fix the byte layout the factory allocates --
while the **label** fields (``quantity_name``, ``dimnames``) are typically
optional, although may be specified or required by a stage. A config may
therefore declare only the structure, leaving labels for the producing
stage to supply; when both producer and consumer expect a label it must agree.

The model is: **buffers carry their own description; stages validate
against it** (and optionally complete any labels the config omitted to minimize
repetition).

Rationale
---------

- Some stages work with arbitrary frame shapes, or move bytes without
  interpreting them, so a buffer's description cannot always come from the
  stages that touch it.
- Validation is the primary safety mechanism: producer, consumer, and
  config descriptions should be cross-checked at startup, so mismatches
  appear at launch (with both descriptions printed) rather than downstream.
- Shapes stay visible in the config, next to the pipeline wiring, for
  review and debugging without consulting stage source.
- Descriptor requirements depend on stage requirements and buffer types:
  ``standard`` buffers carry none, ``ndarray`` buffers carry
  partial descriptors, and ``N2`` buffers carry the full descriptor.

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
  before any data arrives. ``Buffer::ensure_frame_desc`` is the right call for
  shapes derived in code at runtime (e.g. GPU pipeline copy-out) where the
  buffer may or may not be declared: it attaches the descriptor when the buffer
  has none, otherwise reconciles -- validating the structure and completing any
  unset labels. The two differ only in the undeclared case: ``require_frame_desc``
  is fatal when no descriptor is present (config must declare it), whereas
  ``ensure_frame_desc`` attaches one.
