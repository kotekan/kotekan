**************
REST Endpoints
**************

Interaction with a running instance of **kotekan** takes place via a set of REST endpoints.
New processes should adhere to some basic conventions for where to put them and how to interact.

Both ``GET`` and ``POST`` endpoints are supported,
messages to the latter should be formatted as ``json`` strings.

Endpoints should follow a standard structure, to help people locate them:


Framework
**************
The kotekan framework registers a number of points

``/start`` ``[POST]``
    Request an idle instance to begin operation,
    requires a (``json``-encoded) config to be passed.

``/stop`` ``[GET]``
    Tell an active kotekan to shut down and clean up its
    current running configuration.

``/status`` ``[GET]``
    returns the state of the system (active or not).

``/config`` ``[GET]``
    Returns the current system configuration.

``/endpoints`` ``[GET]``
    Returns all available REST endpoints in the system.

``/metrics`` ``[GET]``
    Returns text containing `Prometheus <https://prometheus.io/>`_-formatted
    metrics which serve a host of system state properties.


Per-process
**************
Individual processes can register REST endpoints using code similar to:

.. code-block:: c++

    using namespace std::placeholders;
    restServer * rest_server = get_rest_server();
    // register a POST endpoint
    rest_server->register_json_callback(unique_name + "/my_post_endpoint",
            std::bind(&myKotekanPorcess::endpoint_callback_func, this, _1, _2));
    // register a GET endpoint
    rest_server->register_get_callback(unique_name + "/my_get_endpoint",
            std::bind(&myKotekanPorcess::endpoint_callback_func, this, _1));

Processes should always register endpoints relative to ``/unique_name``.