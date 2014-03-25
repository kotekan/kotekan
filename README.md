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

# Build Requirements

* Standard dev tools and kernel headers

On CentOS:

	yum update
	yum groupinstall "Development tools"
	yum install kernel-headers

* CMAKE 2.6+

* gcc 4.7+

On CentOS 6.5 this requires installing dev tools and entering a different shell:

	wget http://people.centos.org/tru/devtools-1.1/devtools-1.1.repo -O /etc/yum.repos.d/devtools-1.1.repo
	yum install devtoolset-1.1
	scl enable devtoolset-1.1 bash

Once devtools are installed, only the last line needs to be run before building

* AMD Catalyst Drivers (required to use AMD OpenCL)

http://support.amd.com/en-us/download/desktop?os=Linux%20x86_64

Use the latest non-beta drivers, currently 13.12

Install fglrx dependencies:

	yum install compat-libstdc++-33 ld-linux.so.2 libX11.so.6 libXext.so.6 libc.so.6 libdl.so.2 libgcc_s.so.1 libm.so.6 libpthread.so.0 librt.so.1 libstdc++.so.6

Build and install the fglrx RPM:

	./amd-catalyst-13.12-linux-x86.x86_64.run --buildpkg RedHat/RHEL6_64a
	sudo rpm -Uhv fglrx64_p_i_c-13.251-1.x86_64.rpm

Initialize the driver settings:

	sudo amdconfig --initial --adapter=all

* AMD OpenCL SDK

http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/

* PF_RING

The lastest versions of PF_RING seem to have some large API changes, which have not been addressed.
The version that does work is SVN #6818.  Checkout and install that version until the code has
been updated to address the new API

Install process:

	svn co https://svn.ntop.org/svn/ntop/trunk/PF_RING/
	cd PF_RING
	svn up -r 6818


	cd PF_RING/kernel
	make

	(might need to update sim-links)

    Edit PF_RING/drivers/DNA/ixgbe-3.10.16-DNA/src/kcompat.h

	    --- kcompat.h	(revision 6818)
		+++ kcompat.h	(working copy)
		@@ -3129,14 +3129,14 @@
		 #endif
		 
		 /*****************************************************************************/
		-#if ( LINUX_VERSION_CODE < KERNEL_VERSION(3,3,0) )
		-typedef u32 netdev_features_t;
		-#else /* ! < 3.3.0 */
		-#define HAVE_INT_NDO_VLAN_RX_ADD_VID
		-#ifdef ETHTOOL_SRXNTUPLE
		-#undef ETHTOOL_SRXNTUPLE
		-#endif
		-#endif /* < 3.3.0 */
		+//#if ( LINUX_VERSION_CODE < KERNEL_VERSION(3,3,0) )
		+//typedef u32 netdev_features_t;
		+//#else /* ! < 3.3.0 */
		+//#define HAVE_INT_NDO_VLAN_RX_ADD_VID
		+//#ifdef ETHTOOL_SRXNTUPLE
		+//#undef ETHTOOL_SRXNTUPLE
		+//#endif
		+//#endif /* < 3.3.0 */

	cd PF_RING/drivers/DNA/ixgbe-3.10.16-DNA/src/
	make

	cd PF_RING/userland/lib
	./configure --disable-bpf
	make && make install

To load the kernel modules:

	insmod /data/PF_RING/kernel/pf_ring.ko transparent_mode=2 quick_mode=1 enable_tx_capture=0 min_num_slots=4096
	insmod /data/PF_RING/drivers/DNA/ixgbe-3.10.16-DNA/src/ixgbe.ko mtu=16110 RSS=1,1,1,1,1,1,1,1 num_rx_slots=4096 num_tx_slots=0 

To enable the library to be dynamicallyy loaded:

	echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf
	ldconfig

To have everything running at startup add to /etc/rc.local:

	insmod /data/PF_RING/kernel/pf_ring.ko transparent_mode=2 quick_mode=1 enable_tx_capture=0 min_num_slots=4096

	rmmod ixgbe
	insmod /data/PF_RING/drivers/DNA/ixgbe-3.10.16-DNA/src/ixgbe.ko mtu=16110 RSS=1,1,1,1,1,1,1,1 num_rx_slots=4096 num_tx_slots=0

	ldconfig

* HDF5 Development Libraries

* Python 2.7+

On CentOS follow the instructions here to get Python 2.7:
http://toomuchdata.com/2014/02/16/how-to-install-python-on-centos/

You can also install Enthought Canopy if you want to run ch_master on the same system.
If you do install Canopy, you need to run the following cmake command to build the correlator:

	cmake -DPYTHON_INCLUDE_DIRS:PATH=/<Canopy install path>/Canopy/appdata/canopy-1.3.0.1715.rh5-x86_64/include -DPYTHON_EXECUTABLE:FILEPATH=/<Canopy install path>/Canopy/appdata/canopy-1.3.0.1715.rh5-x86_64/bin/python2.7 ..

Where <Canopy install path> is normally the home dir of the user who installed Canopy.>

# Net-to-disk

This proect has not yet been intergrated with cmake, it just has a normal Makefile.

