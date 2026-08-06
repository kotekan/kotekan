ConfigTracker
=============

Overview
--------
``ConfigTracker`` is a singleton service that records the *startup-time* configuration
JSONs of all Kotekan instances participating in a pipeline. The tracker keeps the
node's own *local* config separate from configs received from upstream peers:

- The local config is identified by its JSON content alone (hash over the JSON,
  no host:port). This avoids any dependence on which IP this node decides to
  publish itself under, since a downstream peer may see this node from a
  different address.
- Each upstream entry is keyed by the ``(host, port)`` this node observed when
  it dialed the peer, with the hash baking in that ``(host, port)`` so two
  distinct peers with identical configs still produce distinct hashes.
- Blocks containing ``kotekan_update_endpoint`` are pruned before hashing.

An upstream FPGA controller, when ``/config_tracker/fpga_host_info`` points
at a sibling config block holding ``host`` and ``port``, is registered as a
regular upstream entry: a single combined ``{"config": ..., "timing": ...}``
JSON, keyed by the controller's REST ``(host, port)``. The hostname is
resolved to a canonical IPv4 string up front so downstream peers
transitively land on the same key. Any change to either part after the
initial fetch indicates a controller reset and is fatal. The FPGA snapshot
rides along on the same propagation path as peer kotekan configs, so an
HDF5 writer downstream of the FPGA-adjacent node sees it as an ordinary
upstream entry.

The tracker exposes four REST endpoints:

- ``GET /config_tracker_local`` — returns this node's local ``ConfigInfo``.
- ``GET /config_tracker_local_hash`` — returns ``{"hash": "..."}`` for the local config.
- ``GET /config_tracker_upstream_configs`` — returns the stored upstream configurations
  (optionally filtered by ``hash`` query arg).
- ``GET /config_tracker_upstream_hashes`` — returns the map of upstream config-hash →
  ``{host, port}``.

Why this exists
^^^^^^^^^^^^^^^
Downstream writers can persist the configuration that produced their data.
Instead of shipping full configs with every frame, the sender only flags changes.
The receiver then pulls missing configs via REST and caches them locally using the tracker.

Config block
------------
The tracker is enabled by default. Its behaviour is configured by an optional top-level
``config_tracker`` object (NOT a bare bool — that form is no longer accepted)::

  fpga_controller:
      host: chive.site.chord-observatory.ca
      port: 54321
      config_endpoint: /config              # FPGA controller config path
      timing_endpoint: /get-frame0-time     # FPGA controller timing path

  config_tracker:
      enabled: true                         # default; set false to disable globally
      fpga_host_info: /fpga_controller      # optional; if set, fetch FPGA snapshot at startup
      upstream_fetch_retries: 2             # retries per HTTP request
      upstream_fetch_timeout_seconds: 10    # per-attempt HTTP timeout

The retry/timeout policy applies to every upstream fetch — both the one-shot
FPGA controller fetch at startup and the per-frame peer fetches triggered by
``bufferRecv``'s wire flag. After retries are exhausted on any fetch, the call
is fatal (no silent skip).

The controller's two endpoint paths live on the controller block itself so the
Telescope (which reads ``timing_endpoint`` from the same block via
``gps_host_info`` and uses it as the default for ``gps_endpoint``) and the
ConfigTracker share one source of truth.

Stages that support the tracker (``bufferSend``, ``bufferRecv``) read their per-stage
``use_config_tracker`` first; if unset, they fall back to ``/config_tracker/enabled``;
otherwise they default to ``true``. ``enabled: false`` should be coordinated across the
pipeline — a peer dialing a disabled node sees 404 on the REST endpoints (logged-and-
continued, but noisy).

Tracker-combined-hash
---------------------
The tracker also maintains a **combined MD5** over all stored ``json_hash`` values.
This summarizes the *set* of known configurations on a node. It is inexpensive to compute
and is used by the streaming stages to signal whether configuration state has changed
since the last transmission.

Prometheus metrics
------------------
- ``kotekan_config_tracker_configs_total`` — current number of stored configs (local + upstream).
- ``kotekan_config_tracker_config_present{host,port,hash}`` — labels identify each stored upstream
  ``host:port`` + hash; value is ``1`` while present.
- ``kotekan_config_tracker_local_config_present{hash}`` — labels identify the local config's hash;
  value is ``1`` while present.
- ``kotekan_config_tracker_hash_changes_total`` and
  ``kotekan_config_tracker_last_change_timestamp_seconds`` — change counter and last-change time
  when the combined tracker hash updates or the tracker is reset.
- ``kotekan_config_tracker_upstream_fetch_total{host,port,result="success|fail"}`` — upstream REST
  fetch attempts when pulling configs from peers.

Operational Flow
----------------

1. **Startup registration**
   The local node sets its config via ``setLocalConfig`` as part of kotekan startup, and the
   tracker exposes the REST endpoints.

2. **Sending data**
   A sender (using ``bufferSend``) compares its current tracker-combined-hash to the one last
   sent. If changed, it sets a flag in the frame header (``config_tracker_update = true``) to
   prompt receivers (using ``bufferRecv``) to refresh configuration state.

3. **Receiving data**
   Upon seeing ``config_tracker_update = true``, the receiver calls
   ``getUpstreamConfigs(client_ip, client_port)``. The two-step protocol then:

   1. Fetches ``/config_tracker_local`` from the peer, re-keys that ``ConfigInfo`` under
      the ``(client_ip, client_port)`` actually dialed, and stores it as an upstream entry.
      The dialed address is the validated identity from this node's perspective, regardless
      of how the peer self-named.
   2. Fetches ``/config_tracker_upstream_hashes`` and pulls any missing entries via
      ``/config_tracker_upstream_configs?hash=...``, validating each against its advertised hash.
      FPGA controller snapshots are ordinary upstream entries and ride along on this same path.

   (See the full `doxygen docs <html/>`_ or code for implementation details.)

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
