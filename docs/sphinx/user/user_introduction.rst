************
Introduction
************

So you like the idea of **Kotekan** and want to get started using it? That's what this page is for!

(This assumes you'e got it installed already. If not, see :ref:`compiling`.)

Running kotekan
---------------

**Using systemd (full install)**

To start kotekan

.. code:: bash

    sudo systemctl start kotekan

To stop kotekan

.. code:: bash

    sudo systemctl stop kotekan

**To run in debug mode, run from `ch_gpu/build/kotekan/`**

.. code:: bash

    sudo ./kotekan -c <config_file>.yaml


When installed kotekan's config files are located at /etc/kotekan/

If running with no options then kotekan just stats a rest server, and waits for someone to send it a config in json format on port **12048**.



A Simple Example
----------------
To get things started, let's try running a simple dummy kotekan which generates random numbers, then prints the mean and standard deviation of each frame.

You'll want to create a YAML file containing the relevant config.

We'll make use of two stages, ``generateRandomData`` and ``simpleAutocorr``.

.. todo::
   write these, and make them into a tutorial
