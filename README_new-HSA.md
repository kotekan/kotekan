#A few fixes to enable kotekan on the new HSA driver, which does not have cloc

#Need a newer version of GCC, type
scl enable devtoolset-4 bash

#An example of cmake command without DPDK
cmake -DRTE_SDK=/opt/dpdk-stable-16.11.3/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUS
E_HSA=ON ..
