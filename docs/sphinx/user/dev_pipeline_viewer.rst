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

Starting CPU Monitor
-----------------------
CPU usage is monitored by a separate thread in object CpuMonitor. Thread ids are saved and passed to CPU monitor by stages, 
and CPU monitor keeps tracking all threads every second by reading from proc file system.

To enable CPU usage tracking, add the following to config:
.. code-block:: YAML

    cpu_monitor:
      enabled: true
      track_length: 2  # save last 2 mins cpu usage.

Generating Dump Files
-----------------------
Dump files are generated when kotekan goes down due to errors, but it is possible to manually trigger a 
dump by calling ``dump_trackers()`` from ``KotekanTrackers`` instance.

To enable crash dump, add the following to config:
.. code-block:: YAML

    trackers:
      enable_crash_dump: true
      dump_path: ./

Running Debug Server
-----------------------
Debug server is written in python with Flask, and it provides a way for web interface to fetch run-time data 
from kotekan endpoints and read dump files from file system.

To start the server:
.. code-block:: bash

    export FLASK_APP=debug_server.py FLASK_DEBUG=1
    python debug_server.py

**Note: use option -h to see help info**
"-a=<kotelan_addr>" sets the kotekan address. The default is ``http://localhost:12048``.
"-d=<dump_folder>" sets the dump folder path. The default is ``./``.

To stop the server:
.. code:: bash

    Clt + C

Running Pipeline Viewer
-----------------------
The debug server assumes the kotekan instance is running on the localhost at http://localhost:12048, 
but can be used with a remote host or different port using the ``-a`` flag.

**To show CPU usage, CPU monitor should be enabled in config.**

After starting the debug server, the pipeline viewer is shown here:
``http://localhost:5000/templates/pipeline_tree.html``

Running Dump Viewer
-----------------------
Dump viewer does not need kotekan instance running at the same time, but dump folder is required.
The default path is the current directory but can be set by ``-d`` option.

**To show CPU usage, CPU monitor should be enabled in config.**
**Dump file generating is introduced above.**

After starting the debug server, the dump viewer is shown here:
``http://localhost:5000/templates/dump_viewer.html``
