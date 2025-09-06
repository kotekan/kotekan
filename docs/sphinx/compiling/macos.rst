*********************
MacOS
*********************

MacOS is known to work with most (non-GPU / non-DPDK) options.

You'll need to have `Homebrew <https://brew.sh/>`_ installed, along with Python (we recommend `Anaconda <https://anaconda.org/anaconda/python>`_).

.. code:: bash

    brew install libevent
    pip install pyyaml

To build *docs*, you'll need:

.. code:: bash

    brew install doxygen --with-graphviz
    pip install breathe sphinx_rtd_theme

To include **Airspy** input:

.. code:: bash

    brew install airspy
    cd <kotekan_dir>/build
    cmake .. -DUSE_AIRSPY=ON


To include **FFTW** stages:

.. code:: bash

    brew install fftw
    cd <kotekan_dir>/build
    cmake .. -DUSE_FFTW=ON

To include **OpenSSL** support (for optional support of HASH functions, md5, etc.)

.. code:: bash

    brew install openssl
    cmake -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl ..

To enable **HDF5** writing and linear algebra tools install:

.. code:: bash

    brew install hdf5 lapack openblas
    git clone --recurse-submodules --single-branch --branch v3.1.0 https://github.com/highfive-devs/highfive
    pushd highfive
    cmake -B cmake-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DHIGHFIVE_UNIT_TESTS=OFF \
        -DHIGHFIVE_EXAMPLES=OFF \
        -DHIGHFIVE_BUILD_DOCS=OFF \
        -DHIGHFIVE_TEST_HALF_FLOAT=ON
    cmake --build cmake-build
    cmake --install cmake-build
    popd
    cmake -DUSE_HDF5=ON -DUSE_LAPACK=ON

For a comprehensive build, you'll want

.. code:: bash

    brew install airspy fftw hdf5 lapack



With these installed, you should be able to build & run kotekan!

.. code:: bash

    cd <kotekan_dir>/build
    make
    cd kotekan
    ./kotekan -c `<config_file.yaml>`
