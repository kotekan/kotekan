ConfigTracker
=============

Overview
--------
``ConfigTracker`` is a singleton service that records the *startup-time* configuration
JSONs of all Kotekan instances participating in a pipeline. Each unique
``(host, port)`` pair has exactly one configuration entry, keyed by a canonical
hash of the JSON (with any blocks containing ``kotekan_update_endpoint`` pruned).

The tracker exposes two REST endpoints:

- ``GET /config_tracker_configs`` — returns the stored configurations (optionally filtered by ``hash`` query arg).
- ``GET /config_tracker_hashes`` — returns the map of config-hash → ``host:port``.

Why this exists
^^^^^^^^^^^^^^^
Downstream writers can persist the configuration that produced their data.
Instead of shipping full configs with every frame, the sender only flags changes.
The receiver then pulls missing configs via REST and caches them locally using the tracker.

Config key
----------
The tracker is enabled by default. To disable it globally, set ``config_tracker: false`` in the
*top-level* of your config (not inside a stage block). Stages that support it (e.g. ``bufferSend``,
``bufferRecv``) respect this global setting unless explicitly overridden with a stage-local
``use_config_tracker``.

Tracker-combined-hash
---------------------
The tracker also maintains a **combined MD5** over all stored ``json_hash`` values.
This summarizes the *set* of known configurations on a node. It is inexpensive to compute
and is used by the streaming stages to signal whether configuration state has changed
since the last transmission.

Prometheus metrics
------------------
- ``kotekan_config_tracker_configs_total`` — current number of stored configs.
- ``kotekan_config_tracker_config_present{host,port,hash}`` — labels identify each stored
  ``host:port`` + hash; value is ``1`` while present.
- ``kotekan_config_tracker_hash_changes_total`` and
  ``kotekan_config_tracker_last_change_timestamp_seconds`` — change counter and last-change time
  when the combined tracker hash updates or the tracker is reset.
- ``kotekan_config_tracker_upstream_fetch_total{host,port,result="success|fail"}`` — upstream REST
  fetch attempts when pulling configs from peers.

Operational Flow
----------------

1. **Startup registration**
   The local node registers its config with the tracker as a part of kotekan startup, and 
   the tracker exposes the REST endpoints.

2. **Sending data**
   A sender (using ``bufferSend``) compares its current tracker-combined-hash to the one last
   sent. If changed, it sets a flag in the frame header (``config_tracker_update = true``) to
   prompt receivers (using ``bufferRecv``) to refresh configuration state.

3. **Receiving data**
   Upon seeing ``config_tracker_update = true``, the receiver calls
   ``getUpstreamConfigs(client_ip, client_port)`` to retrieve any missing configs.
   (See the full `doxygen docs <html/>`_ or code for implementation details.)
   The receiver blocks further processing until required configs are present locally.

Threading & Safety
------------------
- All public methods that touch shared state take internal locks.
- The tracker is process-local; network exchange happens via the REST client/servers under the hood.
- Hash collisions are unlikely in practice. If a different hash is found at the same endpoint,
  execution aborts to avoid state contamination.

Per-Connection REST Ports
-------------------------
When receiving frames over the network, ``bufferRecv`` may need to pull upstream configurations
from the sender's REST server (only when the config tracker is enabled).

- Default: the receiver assumes the sender's REST server is on port ``12048`` (``PORT_REST_SERVER``).
- Override: use the stage config key ``upstream_rest_endpoints`` to specify non‑standard ports
  per client. Entries are matched against the client IP as seen by ``bufferRecv``.

Example::

  buffer_recv:
    type: bufferRecv
    listen_port: 11024
    use_config_tracker: true
    upstream_rest_endpoints:
      - "10.1.2.3:13000"
      - "192.168.5.10:14080"

Notes
- This setting is only meaningful when ``use_config_tracker: true``.
- If a client IP:port is not listed, the default port ``12048`` is used for the IP.

Enabling or disabling the tracker
---------------------------------
The config tracker rides along with the streaming stages:

- On the sender, set ``use_config_tracker`` in ``bufferSend`` (default ``true``) to include the
  tracker header and hash updates; set it to ``false`` to omit them.
- On the receiver, set ``use_config_tracker`` in ``bufferRecv`` (default ``true``) to request/
  honour tracker updates; set it to ``false`` if the upstream does not expose the tracker.

Example disabling it on both ends::

  buffer_send:
    type: bufferSend
    use_config_tracker: false

  buffer_recv:
    type: bufferRecv
    use_config_tracker: false
