**************
REST Endpoints
**************

Interaction with a running instance of **kotekan** takes place via a set of REST endpoints.
New stages should adhere to some basic conventions for where to put them and how to interact.

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

``/config_md5sum`` ``[GET]``
    Returns an MD5 hash of the config file (based on the json string with no spaces).
    Only exists if kotekan was build with OpenSSL support included

``/version`` ``[GET]``
    Returns the current kotekan version information; including build options.

``/endpoints`` ``[GET]``
    Returns all available REST endpoints in the system.

``/metrics`` ``[GET]``
    Returns text containing `Prometheus <https://prometheus.io/>`_-formatted
    metrics which serve a host of system state properties.


Per-stage
**************
Individual stages can register REST endpoints using code similar to:

.. code-block:: c++

    using namespace std::placeholders;
    restServer &rest_server = restServer::instance();
    // register a POST endpoint
    rest_server.register_post_callback(unique_name + "/my_post_endpoint",
            std::bind(&myKotekanPorcess::endpoint_callback_func, this, _1, _2));
    // register a GET endpoint
    rest_server.register_get_callback(unique_name + "/my_get_endpoint",
            std::bind(&myKotekanPorcess::endpoint_callback_func, this, _1));

Stages should always register endpoints relative to ``/unique_name``.

The endpoint should be removed in the destructor of the stage registering it:

.. code-block:: c++

    restServer &rest_server = restServer::instance();
    // Remove a GET call back
    rest_server.remove_get_callback(unique_name + "/my_get_endpoint");
    // Remove a POST call back
    rest_server.remove_json_callback(unique_name + "/my_post_endpoint");

Shared Endpoints
*****************
If several stages need to share one endpoint, the endpoint can be created by the `configUpdater`.

.. doxygenclass:: kotekan::configUpdater

Aliases
**************
To make things easier to access, it is possible to define aliases to endpoints in
the config under ``the aliases:`` block in the ``rest_server`` block:

.. code-block:: YAML

    rest_server:
        aliases:
            new_name: existing_endpoint

The above maps ``/new_name`` to ``/existing_endpoint``

The list of aliases available is given by the ``/endpoints`` endpoint.

CPU Affinity
**************
The CPU affinity defaults to the global ``cpu_affinity:`` property

To override that and pin it to say cores 3,4:

.. code-block:: YAML

    rest_server:
        cpu_affinity: [3,4]
