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

       Omitted labels may be supplied by a producing stage (see below).
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
     - none declared; a stage may attach one at runtime via
       ``ensure_frame_desc`` (``require_frame_desc`` would be fatal, as a
       ``standard`` buffer has no declared descriptor)
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
against it.** The buffer factory guarantees the baseline at startup -- a
declared descriptor's byte size matches ``frame_size`` and its structure
matches the config -- so a stage never re-checks the whole shape for that
reason. Beyond that baseline, validation lives in the stage: there is no
shared field-validation helper, because the checks stages need (e.g. dimension
type, buffer size divisibility, etc) are too varied to fold into one. A stage
should read a descriptor and check any specific properties it depends on,
completing any labels the config omitted to minimize repetition.

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

Write ``extents`` as expressions over the genuinely tunable config scalars
(``samples_per_data_set``, ``num_dishes``, ...). Fixed algorithm geometry
(tile and block sizes, packing factors) has a single correct value: prefer
defining it once in code rather than presenting it in config as a tunable.

Choosing how a stage validates a descriptor (from the weakest assertion to the
strongest):

- **Existence only** -- ``Buffer::require_frame_desc()`` (no argument): assert
  the buffer was declared with a descriptor, fatal otherwise. Use it for a stage
  that adapts to whatever shape the config declares, or that depends on only a
  subset of the descriptor; follow it with a typed read and hand-check what you
  actually use.
- **Typed read** -- ``Buffer::require_frame_desc<T>()``: returns the descriptor
  as ``T``, fatal if absent or of the wrong type. It is the safe form of
  ``get_frame_desc<T>()`` (which returns ``nullptr``); use it wherever you would
  immediately dereference the result. This is the idiom for ``N2`` buffers --
  read the ``N2FrameDesc`` and check ``get_num_elements()``, ``get_n2_layout()``,
  ... against what the stage expects.
- **Full cross-check** -- ``Buffer::require_frame_desc(NDArray<T, D>::describe(...))``:
  declare and validate the entire expected ``value_type`` and ``extents``. Use it
  when the stage computes that exact shape from its own config parameters (GPU
  kernels and their CPU counterparts), so a disagreement between the stage's
  parameters and the buffer's declaration is caught at startup.
- **Runtime origination** -- ``Buffer::ensure_frame_desc(describe(...))``:
  attach-or-reconcile, for buffers whose shape is known only at runtime. It
  attaches the descriptor when the buffer has none, otherwise reconciles
  (validating the structure and completing unset labels). It differs from
  ``require_frame_desc`` only in the undeclared case: ``require`` is fatal,
  ``ensure`` attaches.

Whatever the form, prefer a clear ``FATAL_ERROR`` naming the stage and buffer
and printing expected-vs-actual; that is more useful than any generic check.

Avoid:

- re-typing a shape in a stage when the config already declares it and nothing
  in the stage depends on the exact extents -- use the existence-only
  ``require_frame_desc()`` and let the buffer's producer (or another consumer)
  validate the shape;
- a literal ``frame_size`` next to an ``ndarray`` or ``N2`` structure -- the
  size is derived from the descriptor;
- ``ensure_frame_desc`` as a way to tolerate a buffer that should have been
  declared -- that is what ``require``'s fatal is for;
- repeating ``dimnames`` / ``quantity_name`` in config when the producing stage
  already supplies them.

Transmitting descriptors between nodes
--------------------------------------

``bufferSend`` and ``bufferRecv`` can carry a buffer's frame descriptor over the
network so the receiver validates its config-declared descriptor against the
sender's. Enable it with ``use_frame_desc: true`` on **both** stages -- a matched
flag, like ``use_config_tracker`` -- and the wire format is unchanged when it is
off (independent of the config tracker). The descriptor is serialized once per
connection; on receive it is reconciled with ``ensure_frame_desc``, attaching it
to an undeclared (``standard``) buffer or validating a declared one (fatal on
mismatch). This works for every descriptor type, including ``N2``, whose
per-frame metadata carries no structural fields and so cannot otherwise be
validated across the wire.
