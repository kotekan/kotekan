# Build/Run Requirements

## Software:

* CentOS 7.\*, Ubuntu 14.04, 16.04, macOS
* GCC >= 4.9.4 or CLANG >= 3.5.0
* CMake >= 2.8
* libevent, pthread

Required for some build options:

* [DPDK dpdk-16.11.3](http://dpdk.org/)
* Hugepage support
* [AMD OpenCL drivers](http://support.amd.com/en-us/download/linux) and [SDK](http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/)
* [AMD ROCm](https://github.com/RadeonOpenCompute/ROCm)
* [HDF5](https://www.hdfgroup.org/HDF5/) and [HighFive (Richard's fork)](https://github.com/jrs65/HighFive)

## Hardware:

To support the latest HSA builds with full networking stack:

* NIC supporting DPDK, ideally Intel XL710 based
* CPU supporting AVX2, 4 memory channels, and at least 4 real cores. e.g. Intel E5-2620 v3 or i7-5930K
* AMD GPUs R9 Fury (s9300x2), RX 480/580, RX Vega, or later.
* RAM >= 16GB

To support OpenCL builds with the full networking stack:

* NIC supporting DPDK, ideally Intel XL710 based
* CPU supporting AVX2, 4 memory channels, and at least 4 real cores. e.g. Intel E5-2620 v3 or i7-5930K
* AMD GPUs R9 2XX or later.
* RAM >= 16GB

## DPDK

Install process.  Download the DPDK version 16.11.3 from:

http://fast.dpdk.org/rel/dpdk-16.11.3.tar.xz

Unpack it in `/opt/` and run:

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

Cmake build options:

* `-DCMAKE_BUILD_TYPE=Debug` - Builds the project with debug symbols.
* `-DUSE_DPDK=ON -DRTE_SDK=/opt/dpdk-stable-16.11.3/ -DRTE_TARGET=x86_64-native-linuxapp-gcc` - Includes DPDK support.
* `-DUSE_HSA=ON` - Build with HSA support if available. On by default.
* `-DUSE_PRECOMPILED_OPENCL=ON` - For HSA, use precompiled OpenCL .hsaco binaries. Otherwise will attempt cloc.sh to compile.
* `-DUSE_OPENCL=ON` - Build with OpenCL support.
* `-DUSE_HCC=ON` - Build with HCC support, must also set `CXX=hcc`, i.e. `CXX=hcc cmake -DUSE_HCC=ON ..`  This mode has limited support.
* `-DDPDK_VDIF=ON` - Adjusts DPDK defines to optimize for single dish VDIF capture mode.
* `-DUSE_HDF5=ON` and `-DHIGHFIVE_PATH=<path>` - To enable the HDF5 writer
* `-DUSE_AIRSPY=ON` - Build the AirSpy producer. Requires libairspy.
* `-DUSE_FFTW=ON` - Build an FFTW-based F-engine. Requires FFTW3.
* `-DCOMPILE_DOCS=ON` - Build kotekan documentation. Requires doxygen, sphinx, and breathe.

**Examples:**

To build with HSA, DPDK and debug symbols:

    cmake -DRTE_SDK=/opt/dpdk-stable-16.11.3/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON -DCMAKE_BUILD_TYPE=Debug ..

To build with OpenCL and DPDK:

    cmake -DRTE_SDK=/opt/dpdk-stable-16.11.3/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_OPENCL=ON ..

To install kotekan (only works on CentOS at the moment):

	make install

# Running kotekan

**Using systemd (full install)**

To start kotekan

    sudo systemctl start kotekan

To stop kotekan

    sudo systemctl stop kotekan

**To run in debug mode, run from `ch_gpu/build/kotekan/`**

    sudo ./kotekan -c <config_file>.yaml

For example:

    sudo ./kotekan -c ../../kotekan/kotekan_gpu_replay.yaml

When installed kotekan's config files are located at /etc/kotekan/

If running with no options then kotekan just stats a rest server, and waits for someone to send it a config in json format on port `12048`
