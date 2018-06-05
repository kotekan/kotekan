*********************
MacOS
*********************

MacOS is known to work with most (non-GPU / non-DPDK) options.

You'll need to have `Homebrew <https://brew.sh/>`_ installed, along with Python (we recommend `Anaconda <https://anaconda.org/anaconda/python>`_).

.. code:: bash

    brew install libevent
    pip install pyyaml


To include **Airspy** input:

.. code:: bash

    brew install airspy
    cd <kotekan_dir>/build
    cmake .. -DUSE_AIRSPY=ON


To include **FFTW** processes:

.. code:: bash

    brew install fftw
    cd <kotekan_dir>/build
    cmake .. -DUSE_FFTW=ON

To include **OpenSSL** support (for optional support of HASH functions, md5, etc.)

.. code:: bash

    brew install openssl
    cmake -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl ..

With these installed, you should be able to build & run kotekan!

.. code:: bash

    cd <kotekan_dir>/build
    make
    cd kotekan
    ./kotekan -c `<config_file.yaml>`