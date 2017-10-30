# Build/Run Requirements

## Software:

* CentOS 7.*, Ubuntu 14.04, 16.04
* Standard dev tools and kernel headers
* DPDK dpdk-16.11.3
* AMD OpenCL drivers and SDK or ROCm
* GCC >= 4.9.4 or CLANG >= 3.5.0
* Hugepage support

## Hardware:

To support the latest HSA builds with full networking stack:

* NIC supporting DPDK, ideally Intel XL710 based
* CPU supporting AVX2, 4 memory channels, and at least 4 real cores. e.g. Intel E5-2620 v3 or i7-5930K
* AMD GPUs R9 Fury (Nano), RX 480, RX Vega, or later.
* RAM >= 16GB

To support OpenCL builds with the full networking stack:

* NIC supporting DPDK, ideally Intel XL710 based
* CPU supporting AVX2, 4 memory channels, and at least 4 real cores. e.g. Intel E5-2620 v3 or i7-5930K
* AMD GPUs R9 2XX or later.
* RAM >= 16GB

## DPDK

Install process.  Download the DPDK version 2.2.0 from:

http://fast.dpdk.org/rel/dpdk-16.11.3.tar.xz

Unpack it and run:

    make install T=x86_64-native-linuxapp-gcc

It will give a warning about install path being missing, just ingore it.

### Startup scripts to help load DPDK drivers and setup huge pages

Add the following to /etc/rc.local, and make sure rc.local is executable, the following assumes
that Intel XL710 based NIC cards are being used, and dpdk is installed at /data/dpdk-2.2.0/

    echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

    modprobe uio_pci_generic
    PCI_NIC_LIST=`/opt/dpdk-stable-16.11.3/tools/dpdk-devbind.py --status | grep X710 | cut -b 6-12 | tr '\n' ' '`
    /opt/dpdk-stable-16.11.3/tools/dpdk-devbind.py --bind uio_pci_generic $PCI_NIC_LIST

Also add the following line to /etc/fstab to enable huge pages

    nodev   /mnt/huge   hugetlbfs   defaults        0 0

And make the folder /mnt/huge

    sudo mkdir /mnt/huge

# Build Instructions

The project is build using cmake, so you will need to install cmake
before starting a build.

To build just the base framework:

	cd build
	cmake ..
	make

To build with HSA and DPDK change the cmake command to:

    cmake -DRTE_SDK=/opt/dpdk-stable-16.11.3/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON ..

To build with OpenCL and DPDK:

    cmake -DRTE_SDK=/opt/dpdk-stable-16.11.3/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_OPENCL=ON ..

To build with the project with debug symbols add the following to the cmake command `-DCMAKE_BUILD_TYPE=Debug`

To install the program (only works on CentOS at the moment):

	make install

# Running kotekan

**Using systemd (full install)**

To start kotekan

    sudo systemctl start kotekan

To stop kotekan

    sudo systemctl stop kotekan

**To run in debug mode, run from ch_gpu/build/kotekan/**

    sudo ./kotekan -c <config_file>.yaml

When installed kotekan's config files are located at /etc/kotekan/