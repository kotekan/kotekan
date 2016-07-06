# Build/Run Requirements

## Software:

* CentOS 7.*
* Standard dev tools and kernel headers
* DPDK 2.2.0
* AMD OpenCL drivers and SDK
* Jansson
* GCC >= 4.8.5
* Hugepage support

## Hardware:

* NIC supporting DPDK, ideally Intel XL710 based
* CPU supporting AVX2, 4 memory channels, and at least 4 real cores. e.g. Intel E5-2620 v3 or i7-5930K
* AMD GPUs x280 or later, ideally R9 Fury (Nano).
* RAM >= 16GB

Requirements can vary greatly based on required N (number of inputs).

## AMD OpenCL drivers and SDK

* AMD Catalyst Drivers (required to use AMD OpenCL)

Download the RHEL 7 x86_64 rpm file from: http://support.amd.com/en-us/download/

Install fglrx

	sudo yum install fglrx*.rpm

Initialize the driver settings:

	sudo amdconfig --initial --adapter=all

* AMD OpenCL SDK

http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/

Run the .sh file to install the SDK in the default location.

It might be necessary to reboot the computer after installing fglrx, before installing the SDK.

## DPDK

Install process.  Download the DPDK version 2.2.0 from:

http://dpdk.org/browse/dpdk/snapshot/dpdk-2.2.0.tar.gz

Unpack it and run:

    make install T=x86_64-native-linuxapp-gcc

Then export these variables before building the project:

    export RTE_SDK=/<dpdk-location>/
    export RTE_TARGET=x86_64-native-linuxapp-gcc

To make life easy you can add the following to the new file /etc/profile.d/dpdk.sh,
assuming dpdk lives at /data/dpdk-2.2.0/

    export RTE_TARGET=x86_64-native-linuxapp-gcc
    export RTE_SDK=/data/dpdk-2.2.0/

## Install jansson

    sudo yum install jansson*

## Startup scripts to help load DPDK drivers and setup huge pages

Add the following to /etc/rc.local, and make sure rc.local is executable, the following assumes
that Intel XL710 based NIC cards are being used, and dpdk is installed at /data/dpdk-2.2.0/

    echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

    modprobe uio_pci_generic
    PCI_NIC_LIST=`/data/dpdk-2.2.0/tools/dpdk_nic_bind.py --status | grep X710 | cut -b 6-12 | tr '\n' ' '`
    /data/dpdk-2.2.0/tools/dpdk_nic_bind.py --bind uio_pci_generic $PCI_NIC_LIST

Also add the following line to /etc/fstab to enable huge pages

    nodev   /mnt/huge   hugetlbfs   defaults        0 0

And make the folder /mnt/huge

    sudo mkdir /mnt/huge

# Build Instructions

The project is build using cmake, so you will need to install cmake
before starting a build.

To build everything:

	cd build
	cmake ..
	make

The binaries will be located in subdirs mirroring the source tree.
For example the correlator binary will be in build/correlator/

To just build one project:

	cd build
	cmake ..
	cd <project you want to build>
	make

This will build just that project plus any dependencies it has.

To build with the project with debug symbols:

	cd build
	cmake -DCMAKE_BUILD_TYPE=Debug ..
	make

To install the program (only works on CentOS at the moment):

	make install

# Running kotekan

* Using systemd (full install)

To start kotekan

    sudo systemctl start kotekan

To stop kotekan

    sudo systemctl stop kotekan

* To run in debug mode, run from ch_gpu/build/kotekan/

    sudo ./kotekan -c ../kotekan/kotekan.conf -l 3 -s

When installed kotekan's config files are located at /etc/kotekan/