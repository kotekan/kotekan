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
- ``/pipeline_dot`` (GET) – the running pipeline graph in graphviz ``dot`` format.
- ``/buffer/<name>/frame`` (GET) – copy of the newest full frame in buffer ``<name>`` as JSON:
  base64 ``data``, the frame's metadata, and the buffer's frame descriptor. ``?len=N`` limits
  the number of data bytes returned; ``len=0`` returns metadata only.
- Per-stage endpoints live under the stage ``unique_name`` (e.g., ``/<stage>/control``).

Example:

.. code-block:: bash

    curl -X POST -H "Content-Type: application/json" --data @config.json \
      http://localhost:12048/start


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
