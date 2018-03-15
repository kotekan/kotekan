..  _openblas:

=======================
Configuring OpenBLAS
=======================

Optional and only required for some kotekan processes. Only from-source builds are supported, not the
version that is distributed with Debian/Ubuntu.

Obtain the source from the OpenBLAS homepage or via:

    git clone https://github.com/xianyi/OpenBLAS.git

Enter the source directory and run:

    make
    make PREFIX=<openblas_prefix> install

where ``<openblas_prefix>`` is the *full* path to a directory of your choice, eg. ``/opt/OpenBLAS``, or just a
``build`` directory in the source dir. Set the environment variable ``CMAKE_PREFIX_PATH`` to include
``<openblas_prefix>`` or supply ``-DOPENBLAS_PATH=<openblas_prefix>`` to ``cmake`` when building kotekan.

