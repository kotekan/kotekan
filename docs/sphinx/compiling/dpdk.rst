..  _dpdk:

================
Configuring DPDK
================

Install
^^^^^^^

To install DPDK on Ubuntu 22.04 install run

.. code:: bash

     sudo apt install dpdk dpdk-dev dpdk-kmods-dkms

Then CMake will automatically include it. Note that DPDK
is not compatible with BOOST tests, so will be disabled if
the ``-DWITH_BOOST_TESTS`` option is enabled.

To install DPDK on ubuntu 18.04 simply run:

.. code:: bash

     sudo apt install dpdk dpdk-dev dpdk-igb-uio-dkms

Then include ``-DUSE_OLD_DPDK=ON`` in the cmake options.


Setup
^^^^^

Run ``dpdk-devbind --status`` to see the addresses of your system NICs, then
add the NICs you want to have bound to DPDK at startup to ``/etc/dpdk/interfaces``
for example to bind two Intel XL710 NICs to the ``uio_igb`` driver at PCIe addresses
``0000:46:00.0`` and ``0000:48:00.0`` add the following two lines:

.. code:: text

    pci 0000:46:00.0    igb_uio
    pci 0000:48:00.0    igb_uio

Then edit ``/etc/dpdk/dpdk.conf`` and add the following line:

.. code:: text

    NR_2M_PAGES=1024

Reboot the system and check ``dpdk-devbind --status`` to see if the NICs were correctly
bound to the UIO drivers.

Please note that depending on the setup you may want to use more or less huge pages.
If you have a NUMA system you'll need to manually create your huge pages on each NUMA node,
the above script will not work.

============================
Configuring DPDK from source
============================

To build from source
^^^^^^^^^^^^^^^^^^^^

This is not required for Ubuntu systems which package DPDK

Download: `DPDK version 16.11.10 <http://fast.dpdk.org/rel/dpdk-16.11.10.tar.xz>`_

Unpack it in ``/opt/`` and run:

.. code:: bash

    sudo make install T=x86_64-native-linuxapp-gcc

It will give a warning about install path being missing, just ingore it.

When running CMAKE include the following options ``-DUSE_OLD_DPDK=ON -DRTE_SDK=/opt/dpdk-stable-16.11.10/ -DRTE_TARGET=x86_64-native-linuxapp-gcc``

Startup scripts source install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To automatically load DPDK drivers and setup huge pages:

Add the following to ``/etc/rc.local``, and make sure ``rc.local`` is executable. (This code assumes that Intel XL710 based NIC cards are being used, and dpdk is installed at ``/opt/dpdk-stable-16.11.9/``, adjust as appropriate.)

.. code:: bash

    echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

    modprobe uio
    insmod /opt/dpdk-stable-16.11.10/x86_64-native-linuxapp-gcc/kmod/igb_uio.ko
    PCI_NIC_LIST=`/opt/dpdk-stable-16.11.10/tools/dpdk-devbind.py --status | grep X710 | cut -b 6-12 | tr '\n' ' '`
    /opt/dpdk-stable-16.11.10/tools/dpdk-devbind.py --bind igb_uio $PCI_NIC_LIST

You may need to adjust ``X710`` to ``XL710`` above depending on your NIC.

Also add the following line to /etc/fstab to enable huge pages

.. code:: bash

    nodev   /mnt/huge   hugetlbfs   defaults        0 0

And make the folder /mnt/huge

.. code:: bash

    sudo mkdir /mnt/huge
