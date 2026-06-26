.. _user_config:

*************
Configuration
*************

**Kotekan** configurations can hold a large number of stages and system configuration settings.
Below, some options are documented.
A typical config has top-level sections for the REST server, metadata pools,
buffers, and stages. The dataset manager may also have a section, but is being removed in CHORD
in favor of the Config Tracker.
Names are referenced by their YAML path
in stage settings (e.g., ``in_buf``/``out_buf`` or metadata pool references).

Buffers
-------

Buffers are declared anywhere in the YAML tree by adding ``kotekan_buffer`` to a block. The block
name (and its YAML path) is what stages reference in their ``in_buf``/``out_buf`` settings.
Typed buffers (``vis``, ``N2``, ``hfb``) need a compatible metadata pool; ``standard`` buffers can
omit metadata entirely.


.. code-block:: yaml

    vis_pool:
      kotekan_metadata_pool: VisMetadata
      num_metadata_objects: 16

    vis_buffer:
      kotekan_buffer: vis
      num_frames: 8
      metadata_pool: vis_pool
      numa_node: 0
      zero_new_frames: true

Common keys (unless noted otherwise):

- ``kotekan_buffer``: buffer type; supported values are ``standard``, ``ndarray``, ``vis``, ``N2``,
  ``hfb``, and ``ring``.
- ``num_frames``: depth of frame-based buffers (everything except ``ring``). Pick a size that covers
  the latency of downstream stages.
- ``metadata_pool``: optional for ``standard`` buffers; name of a :ref:`metadata pool
  <metadata_config>` block. If omitted, the buffer carries no metadata.
- ``numa_node`` (default ``0``), ``use_hugepages`` (default ``false``), ``mlock_frames`` (default
  ``true``), ``zero_new_frames`` (default ``true``), ``cpu_affinity`` (pin the zeroing helper
  threads), and ``log_level`` mirror the values on stages. ``zero_new_frames`` only zeros memory
  when it is first allocated; reused frames are not cleared unless a stage calls ``zero_frames()``
  or explicitly writes over the data.

Type-specific notes:

- ``standard`` – raw byte buffers. You must set ``frame_size`` in bytes.
- ``ndarray`` – frames holding a single typed multi-dimensional array. The **required** structural
  fields are ``value_type`` (a kotekan ``DataType`` name such as ``float32`` or ``int4x2``) and
  ``extents`` (a list of dimension sizes; entries may be arithmetic expressions referencing other
  config values, e.g. ``samples_per_data_set / upchan_factor``) -- these fix the byte layout the
  factory allocates. The semantic labels ``quantity_name`` and ``dimnames`` (a list of axis labels,
  one per extent) are **optional**: when omitted, the producing stage supplies them; when given,
  the stage validates against them. ``frame_size`` is derived from the descriptor, and the frame
  descriptor is attached to the buffer at startup, so stages can validate against it (or read
  shapes from it) from their constructors onward. Typically paired with a ``chordMetadata`` pool.
  See :ref:`dev_buffers` for the descriptor model and authoring guidance.
- ``vis`` – visibility frames sized automatically from ``num_elements``, ``num_ev``, and optional
  ``num_prod`` (defaults to an upper-triangular visibility matrix). Numeric types are fixed
  (complex floats for visibilities/EVs, floats for weights/flags). The attached metadata type is
  ``VisMetadata``.
- ``N2`` – CHORD N2 visibility frames sized from ``num_elements``, ``num_ev``, and ``vis_layout``
  (see ``N2Layout`` in :mod:`kotekan.n2k`). Numeric types are fixed (complex floats and floats for
  ancillary fields). Uses ``N2Metadata``.
- ``hfb`` – hyperfine beam buffers sized from ``num_frb_total_beams`` and ``factor_upchan``. Numeric
  types are fixed (floats). Uses ``HFBMetadata``.
- ``ring`` – coordination-only buffer for external GPU/FPGA ring memory. Use ``ring_buffer_size``
  instead of ``num_frames``; no host-side frames are allocated.


.. _metadata_config:


Metadata
--------

Metadata is attached per frame and reference-counted as it moves between buffers. It is only
available if the buffer points at a metadata pool.

Declaring pools:

.. code-block:: yaml

    vis_pool:
      kotekan_metadata_pool: VisMetadata
      num_metadata_objects: 16

- ``kotekan_metadata_pool`` selects the type (see list below).
- ``num_metadata_objects`` is the intended number of metadata objects to keep in circulation. Set it
  to at least the number of frames that can be in flight (e.g., the largest buffer depth using this
  pool). Kotekan currently allocates objects on demand but still records this size for the pool.

Attaching pools:

- Add ``metadata_pool: <pool name>`` on a buffer so producers can allocate metadata for its frames.
  Without it, metadata helpers no-op and dataset IDs cannot be recorded for that buffer.
- Producers typically allocate metadata for new frames; stages that forward frames should pass
  metadata along rather than creating new objects so dataset IDs and timing fields are preserved.

Available metadata types:

- ``VisMetadata`` – visibility streams; carries dataset ID, frequency, FPGA sequence counters, and
  dimensions (``num_elements``/``num_prod``/``num_ev``).
- ``N2Metadata`` – N2 visibility streams; adds absolute time indices, Earth-orientation parameters,
  and dataset ID for CHORD N2 pipelines.
- ``HFBMetadata`` – hyperfine beam outputs; dataset ID, frequency, FPGA sequence counters, beam and
  sub-frequency counts.
- ``BeamMetadata`` – tracked beamformer outputs; dataset ID plus pointing (RA/Dec), beam number, and
  coarse frequency list.
- ``BasebandMetadata`` – baseband capture; event IDs, frequency and sequence ranges, timestamps per
  event, and input count.
- ``chordMetadata`` – generalized ND-array metadata for CHORD/PFB paths; includes dataset IDs when
  present along with array layout, time/frequency information, and telescope geometry.
- ``oneHotMetadata`` – lightweight metadata for synthetic one-hot tests (frame counter and indices).

Dataset IDs are stored by most science metadata types above and are managed by the dataset manager,
so pools should be defined before any buffer that needs dataset-aware processing.


Dataset manager & config tracker
--------------------------------

- The dataset manager tracks dataset IDs and their states across stages, and is typically used in CHIME
pipelines. Commonly used config parameters include (under ``dataset_manager``):
  - ``use_dataset_broker`` (bool): sync states with the central broker (comet); must be set when
    broker coordination is desired.
  - ``ds_broker_host`` / ``ds_broker_port``: broker address/port (default 127.0.0.1:12050 when
    provided).
  - ``retry_wait_time_ms``, ``retries_rest_client``, ``timeout_rest_client_s``: broker REST retry/timing.
  - ``force_dataset_broker_update`` (optional): force registration on startup.
  - REST: ``/dataset-manager/force-update`` forces re-registration with the broker.
  - Metrics: ``kotekan_datasetbroker_error_count`` (broker comms errors), plus per-stage dataset
    metrics (see writer/transform stages).
- The Config Tracker is intended as a more lightweight replacement for the dataset manager.
  It stores snapshots of this node's own (local) config and configs from upstream peers, and
  optionally a combined ``{"config": ..., "timing": ...}`` snapshot fetched from an upstream
  FPGA controller. All entries propagate to downstream nodes over REST. Configured by an
  optional top-level ``config_tracker`` block::

    fpga_controller:
        host: chive.site.chord-observatory.ca
        port: 54321
        config_endpoint: /config                  # consumed by ConfigTracker
        timing_endpoint: /get-frame0-time         # consumed by Telescope + ConfigTracker
        gains_endpoint: /get-current-gain-file    # consumed by hdf5N2Write

    config_tracker:
        enabled: true                          # default; false disables the tracker globally
        fpga_host_info: /fpga_controller       # optional; names the controller block
        upstream_fetch_retries: 2              # retries per HTTP request (FPGA + peer pulls)
        upstream_fetch_timeout_seconds: 10     # per-attempt HTTP timeout (FPGA + peer pulls)

  The ``fpga_host_info`` indirection lets a pipeline define the FPGA controller address and
  endpoints once and have multiple consumers (the Config Tracker plus the Telescope's
  ``gps_host_info`` and ``hdf5N2Write``'s ``baseband_gain_host_info``) point at it. The
  Telescope picks up ``timing_endpoint`` from the same block as the default for its own
  ``gps_endpoint``, and ``hdf5N2Write`` picks up ``gains_endpoint`` (default
  ``/get-current-gain-file``) for the startup gains-file fetch, so paths aren't duplicated.
  Stages that support the tracker (``bufferSend``/``bufferRecv``) read their per-stage
  ``use_config_tracker`` first; if unset, they fall back to ``/config_tracker/enabled``.

  You can use the ``configTrackerWriter`` stage with:

  - ``base_dir``: directory to write JSON snapshots (created if missing).
  - Output naming: ``local.json`` for the local entry, ``<host>_<port>.json`` for every
    upstream entry (including the FPGA controller snapshot).
  - Files are rotated on tracker-hash change.


Stages
------

.. toctree::
    :glob:
    :maxdepth: 1

    processes/*


Variable Evaluation
-------------------

Config values are resolved by walking up the YAML path: a stage first looks in its own block, then
parent scopes, then the root. Jinja2 templating can be used in ``.j2`` configs; supply variables via
``-e '{"key": "value"}'`` when invoking kotekan. Shared constants can be defined at higher levels
and referenced by name in child blocks.

Config usage tracking
---------------------

Set the top-level ``log_config_usage: true`` to have kotekan record which config leaf items (scalar
values, strings, and lists -- not the object blocks that contain them) are actually read while a
pipeline runs, and by which path. At shutdown an ``INFO`` summary is logged listing the accessed
items (with the requesting base paths, typically stage ``unique_name``\ s) and the items that were
never accessed. This is useful for spotting stale or misspelled config keys that no stage consumes.
The feature is off by default and adds no overhead unless enabled.

Framework dispatch markers (``kotekan_buffer``, ``kotekan_stage``, ``kotekan_metadata_pool``,
``kotekan_update_endpoint``) are excluded from the summary: the factories and config updater consume
those by walking the config tree directly rather than through a value lookup, so they would otherwise
always appear as unused.

REST server
-----------

The ``rest_server`` block controls binding and convenience aliases. Bind to an IPv4 address and port
with ``bind_address``/``port`` (defaults to ``0.0.0.0`` and ``12048``), set ``cpu_affinity`` to pin
the REST thread, and use ``aliases`` to map short paths to full endpoints (listed under
``/endpoints``).
