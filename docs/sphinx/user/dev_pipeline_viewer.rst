****************
Pipeline Viewer
****************

Pipeline viewer is a debug tool to visualize pipeline structure and node utilizations.

Buffers and stages are using round and rectangular boxes respectively.
Buffer frames and stage CPU usages are displayed under their buffer or stage names.
The graph is zoomable and draggable, and utilizations are updated dynamically.

Running Pipeline Viewer
-----------------------
Start kotekan **first** under ``/build/kotekan/``

.. code:: bash

    ./kotekan -c <path_to_config_file>

Then start debug server under ``/tools/debug_tool/``

.. code:: bash

    export FLASK_APP=debug_server.py FLASK_DEBUG=1
    python debug_server.py

**Note: kotekan address can be specific by adding "-a=<kotelan_addr>" option. The default is ``"http://localhost:12048"``.**

The generated graph is shown here:
``http://localhost:5000/templates/pipeline_tree.html``

To stop debug server and kotekan

.. code:: bash

    Clt + C

