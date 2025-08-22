ConfigTracker
=============

Overview
--------
``ConfigTracker`` is a singleton service that records the *startup-time* configuration
JSONs of all Kotekan instances participating in a pipeline. Each unique
``(host, port)`` pair has exactly one configuration entry, keyed by a canonical
hash of the JSON (with any ``updatable_config`` pruned).

The tracker exposes two REST endpoints:

- ``GET /config_tracker_configs`` — returns the stored configurations (optionally filtered by ``hash`` query arg).
- ``GET /config_tracker_hashes`` — returns the map of config-hash → ``host:port``.

Why this exists
^^^^^^^^^^^^^^^
Downstream writers can persist the configuration that produced their data.
Instead of shipping full configs with every frame, the sender only flags changes.
The receiver then pulls missing configs via REST and caches them locally using the tracker.

Tracker-combined-hash
---------------------
The tracker also maintains a **combined MD5** over all stored ``json_hash`` values.
This summarizes the *set* of known configurations on a node. It is inexpensive to compute
and is used by the streaming stages to signal whether configuration state has changed
since the last transmission.

Operational Flow
----------------

1. **Startup registration**
   The local node registers its full filtered config with the tracker and exposes the REST endpoints.

2. **Sending data**
   The sender compares its current tracker-combined-hash to the one last sent. If changed,
   it sets a flag in the frame header (``config_tracker_update = true``) to prompt receivers
   to refresh configuration state.

3. **Receiving data**
   Upon seeing ``config_tracker_update = true``, the receiver calls
   ``getUpstreamConfigs(client_ip, client_port)`` to retrieve any missing configs by hash.
   The receiver blocks further processing until required configs are present locally.

Threading & Safety
------------------
- All public methods that touch shared state take internal locks.
- The tracker is process-local; network exchange happens via the REST client/servers under the hood.
- Hash collisions are unlikely in practice; if the same hash is mapped to different endpoints,
  execution aborts to avoid misattribution.
