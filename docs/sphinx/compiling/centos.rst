*********************
CentOS
*********************

CentOS works well with most kotekan options, but requires ``devtoolset-4`` or greater.

You'll need to have Python installed (we recommend `Anaconda <https://anaconda.org/anaconda/python>`_).

.. code:: bash

    sudo yum install centos-release-scl
    sudo yum install devtoolset-4
    sudo yum install libevent-devel
    sudo yum install python-yaml

Before running ``cmake`` or building kotekan, you'll have to switch to ``devtoolset-4``.

.. code:: bash

    scl enable devtoolset-4 bash

You should now be able to build & run kotekan!

.. code:: bash

    cd <kotekan_dir>/build
    make
    cd kotekan
    ./kotekan -c `<config_file.yaml>`


Airspy
--------
To include **Airspy** input, you'll have to roll your own RPM.
We suggest following `these instructions <http://www.trinityos.com/HAM/CentosDigitalModes/hampacketizing-centos.html#42g.airspy-sw>`_. Briefly:

.. code:: bash

    sudo yum install rpm-build
    cd ~/rpmbuild/SOURCES
    wget https://github.com/airspy/host/archive/v1.0.8.tar.gz
    mv v1.0.8.tar.gz airspyone_host-1.0.8.tar.gz
    cd ../SPECS
    wget www.trinityos.com/HAM/CentosDigitalModes/usr/src/redhat/SPECS/airspy-host.spec
    <edit the .spec file, change the Name: entry to "airspyone_host">
    rpmbuild -bb --target=x86_64 airspy-host.spec
    sudo yum install ../RPMS/x86_64/airspyone_host-1.0.8-1.el6.x86_64.rpm

You should now be able to build with airspy,

.. code:: bash

    cd <kotekan_dir>/build
    cmake .. -DUSE_AIRSPY=ON


FFTW
-----
To include **FFTW** processes:

.. code:: bash

    sudo yum install fftw3
    cd <kotekan_dir>/build
    cmake .. -DUSE_FFTW=ON

