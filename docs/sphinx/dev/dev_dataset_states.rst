****************************
Management of Dataset States
****************************

A *dataset* is describes a set of data with consistent metadata. It is uniquely
identified by its ID and is defined by the ID of a *base dataset* and an ID of
the *datasetState* that describes the differences between it and its *base
dataset*. A *datasetState* is describing the metadata of a *dataset* and is
uniquely identified by its ID. Synchronizing both *datsetStates* and the
topologies of *datasets* is done by the *datasetManager*.
The latter exists as a singleton in every kotekan instance that uses it in its
configuration. If configured, different kotekan instances can synchronize their
*datasets* and *datasetStates* via the zentralized *dataset_broker*. The
communication protocol is documented in :ref:`dataset-broker`.

.. doxygenclass:: datasetManager
    :members:

.. doxygenclass:: dataset
    :members:

.. doxygenclass:: datasetState
    :members:

.. doxygenclass:: freqState
    :members:

.. doxygenclass:: inputState
    :members:

.. doxygenclass:: prodState
    :members:

.. doxygenclass:: stackState
    :members:

.. doxygenclass:: timeState
    :members:

.. doxygenclass:: metadataState
    :members:

.. doxygenclass:: eigenvalueState
    :members: