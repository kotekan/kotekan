************
Operation
************

REST endpoints
--------------

- ``/start`` (POST JSON config) – start a pipeline from a posted config.
- ``/stop`` (GET) – stop the running pipeline.
- ``/status`` (GET) – returns whether kotekan is running.
- ``/config`` (GET) – returns the active config.
- ``/version`` (GET) – version/build info.
- ``/endpoints`` (GET) – list all registered endpoints (including aliases).
- ``/metrics`` (GET) – Prometheus-format metrics (buffers, stages, broker errors, etc.).
- ``/buffers`` (GET) – per-buffer status: frame fullness, frame size, last arrival time.
- ``/pipeline_dot`` (GET) – the running pipeline graph in graphviz ``dot`` format
  (``Content-Type: text/vnd.graphviz``). See `Pipeline graph`_.
- ``/pipeline_json`` (GET) – the same graph as JSON (nodes, edges, clusters), for clients
  that would rather lay it out, or diff it, than parse ``dot``. Takes the same query
  arguments.
- ``/buffer/<name>/frame`` (GET) – copy of the newest full frame in buffer ``<name>`` as JSON:
  base64 ``data``, the frame's metadata, and the buffer's frame descriptor. ``?len=N`` limits
  the number of data bytes returned; ``len=0`` returns metadata only. On buffers whose
  consumers drain frames faster than a peek can catch them, set ``peek_hold: true`` on the
  buffer's config block: recycling of the newest frame is then deferred until the next one
  lands (no data is copied), so this endpoint always has a frame to serve. The held frame can
  be arbitrarily old if production has stopped — check the metadata timestamps — and it counts
  as one full frame in ``/buffers``.
- Per-stage endpoints live under the stage ``unique_name`` (e.g., ``/<stage>/control``).

Example:

.. code-block:: bash

    curl -X POST -H "Content-Type: application/json" --data @config.json \
      http://localhost:12048/start


Pipeline graph
--------------

``/pipeline_dot`` draws the pipeline as it is actually running, so it shows things a
drawing made from the config cannot:

- Producer and consumer directions are the registrations themselves, not a guess from
  the config key names.
- A buffer's array layout comes from its frame descriptor: the shape the data has, not
  one inferred from the config or the kernel sources.
- Measured frame and byte rates, fill levels, CPU usage per stage, and GPU kernel times
  (the last needs ``profiling: true``, CPU usage needs ``/cpu_monitor: {enabled: true}``).
- A buffer with no empty frame left is outlined in red — that is where the pipeline is
  backed up — and a buffer nothing has ever flowed through is dashed.

Stages are grouped into the config section they were declared in, each GPU device is a
region holding its commands and device memory, and a buffer is drawn inside the innermost
region containing everything that touches it, so only real hand-offs cross a boundary.
Buffer nodes link to their ``/buffer/<name>/frame`` endpoint, so a rendered SVG is
clickable.

Every colour in the graph is named explicitly, including the label ink and the edges that
graphviz would otherwise leave black. That is what lets a viewer restyle the whole graph —
choco's pipeline page mirrors the palette to draw it on a dark background — so the values
in ``lib/core/PipelineGraph.cpp`` (``graph_style()`` and the ``graph_*`` constants) are an
interface, not a local choice: changing one changes what those viewers have to match.

Query arguments (all boolean ones take ``0``/``1``):

- ``rankdir=LR|TB|RL|BT`` – layout direction (default ``LR``).
- ``cluster=0`` – do not group stages by config section.
- ``legend=0`` / ``pools=0`` – drop the colour key / the metadata pool list.
- ``kernels=0`` – hide GPU commands and device memory; useful on large pipelines where
  they dominate the graph.
- ``runtime=0`` – structure only, no live numbers or state colouring. Use this to compare
  two runs: the live figures differ on every fetch.
- ``urls=0`` – no links on buffer nodes.

.. code-block:: bash

    # render the running pipeline
    curl -s http://localhost:12048/pipeline_dot | dot -Tsvg -o pipeline.svg

    # just the structure, without the GPU internals
    curl -s 'http://localhost:12048/pipeline_dot?runtime=0&kernels=0' | dot -Tpdf -o pipeline.pdf

``tools/pipeline_graph.py`` wraps this up: it fetches, renders, and can watch a running
pipeline by re-fetching on an interval.


Daemon mode
-------------

Running kotekan without ``-c`` starts only the REST server and waits for ``/start``. Bind with
``-b <ipv4:port>`` (defaults to ``0.0.0.0:12048``). In this mode kotekan logs to syslog by default;
add ``-n`` to suppress stderr, or ``-s`` to force syslog when running with a config file.


Monitoring
-----------

Use ``/metrics`` with Prometheus to scrape buffer fill/drop rates, stage timing, and dataset broker
error counts. The pipeline viewer can be pointed at a running instance for a live graph, and the
``bufferStatus`` stage can export buffer depths via the metrics endpoint.


Config tracker and dataset broker
----------------------------------

If ``configTrackerWriter`` is enabled, configs are snapshotted to disk for auditing; hashes and
records are also exposed via the tracker endpoints listed under ``/endpoints``. When using the
dataset broker, ``/dataset-manager/force-update`` forces re-registration of datasets and states.


Useful Modules
---------------
.. toctree::
    :glob:
    :maxdepth: 1

    modules/*
