*********************
Ubuntu
*********************

Ubuntu works well with all options.

You'll need to have Python installed (we recommend `Anaconda <https://anaconda.org/anaconda/python>`_).

.. code:: bash

    sudo apt-get install libevent-dev
    sudo apt-get install python-yaml


To include **Airspy** input:

.. code:: bash

    sudo apt-get install airspy-dev
    cd <kotekan_dir>/build
    cmake .. -DUSE_AIRSPY=ON


To include **FFTW** stages:

.. code:: bash

    sudo apt-get install fftw3
    cd <kotekan_dir>/build
    cmake .. -DUSE_FFTW=ON


With these installed, you should be able to build & run kotekan!

.. code:: bash

    cd <kotekan_dir>/build
    cmake <options> ..
    make
    cd kotekan
    ./kotekan -c `<config_file.yaml>`
