..  _dpdk:

================
Configuring DPDK
================

(Instructions for Linux x86_64 hosts.)

Download: `DPDK version 16.11.4 <http://fast.dpdk.org/rel/dpdk-16.11.4.tar.xz>`_

Unpack it in ``/opt/`` and run:

.. code:: bash

    make install T=x86_64-native-linuxapp-gcc

It will give a warning about install path being missing, just ingore it.

Startup scripts
^^^^^^^^^^^^^^^

To automatically load DPDK drivers and setup huge pages:

Add the following to ``/etc/rc.local``, and make sure ``rc.local`` is executable. (This code assumes that Intel XL710 based NIC cards are being used, and dpdk is installed at ``/opt/dpdk-stable-16.11.4/``, adjust as appropriate.)

.. code:: bash

    echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

    modprobe uio
    insmod /opt/dpdk-stable-16.11.4/x86_64-native-linuxapp-gcc/kmod/igb_uio.ko
    PCI_NIC_LIST=`/opt/dpdk-stable-16.11.4/tools/dpdk-devbind.py --status | grep X710 | cut -b 6-12 | tr '\n' ' '`
    /opt/dpdk-stable-16.11.4/tools/dpdk-devbind.py --bind igb_uio $PCI_NIC_LIST

Also add the following line to /etc/fstab to enable huge pages

.. code:: bash

    nodev   /mnt/huge   hugetlbfs   defaults        0 0

And make the folder /mnt/huge

.. code:: bash

    sudo mkdir /mnt/huge
