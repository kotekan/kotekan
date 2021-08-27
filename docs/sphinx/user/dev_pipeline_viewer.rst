****************
Pipeline Viewer
****************

Pipeline viewer is a debug tool to visualize real-time pipeline structure and node utilization.

Buffers and stages are using round and rectangular boxes respectively.
Buffer frames and stage CPU usages are displayed under their buffer or stage names.
Each stage node also shows its first two trackers with current value and unit.
The graph is zoomable and draggable, and utilizations are updated dynamically.

A sidebar is on the right to give more details, and clicking on stage labels will trigger 
the tracker table shown in the sidebar. Every tracker table includes all trackers under the clicked 
stage and also the CPU usage of the user and system mode. All stats are provided in one row, in the 
order of tracker name, current value, unit, avg, std, min, max.

****************
Dump Viewer
****************

Dump viewer is a tool to visualize pipeline structure and node stats from dump files.

functionalities are similar to pipeline viewer, but it is static and not updated automatically.
It is convenient to view historical tracker info based on a selected time.

On the left, there are a button called "Select a file" and a time slider. The button gives file choices 
from the given dump folder that specified in debug_server, and selecting any of them will trigger an update 
to the graph and sidebar. The time slider is used to choose a time point, and all stats and usage numbers will 
be updated to the latest ones just before that selected time.

In the sidebar, the only difference from pipeline viewer is that dump viewer shows the timestamp of each "current" value.

Running Pipeline Viewer
-----------------------
Start kotekan **first** under ``/build/kotekan/``

.. code:: bash

    ./kotekan -c <path_to_config_file>

Then start debug server under ``/tools/debug_tool/``

.. code:: bash

    export FLASK_APP=debug_server.py FLASK_DEBUG=1
    python debug_server.py

**Note: use option -h to see help info**
"-a=<kotelan_addr>" sets the kotekan address. The default is ``http://localhost:12048``.
"-d=<dump_folder>" sets the dump folder path. The default is ``./``.

The pipeline viewer is shown here:
``http://localhost:5000/templates/pipeline_tree.html``

The dump viewer is shown here:
``http://localhost:5000/templates/dump_viewer.html``

To stop debug server and kotekan

.. code:: bash

    Clt + C

