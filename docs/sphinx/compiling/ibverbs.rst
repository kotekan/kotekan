..  _ibverbs:

===================
Configuring ibverbs
===================

The ``ibvUdpSend`` and ``ibvUdpRecv`` stages send and receive UDP packets through raw packet
queue pairs on NVIDIA/Mellanox ConnectX NICs (ConnectX-6 and ConnectX-7) using ``libibverbs``.
The NIC reads and writes kotekan frames directly: packet headers go to one buffer and payloads to
another, and the CPU never copies a packet. ``udpPacketGen`` and ``udpPacketCheck`` generate and
verify test traffic for them.

Install
^^^^^^^

.. code:: bash

     sudo apt install libibverbs-dev ibverbs-providers rdma-core

CMake enables the stages when ``pkg-config`` finds ``libibverbs``. Use ``-DUSE_IBVERBS=ON`` to
require it or ``OFF`` to leave the stages out. The NIC stays bound to its kernel driver: no DPDK
style rebinding is needed, and the interface keeps its IP address, so the peer can resolve its
MAC with ARP as usual.

Setup
^^^^^

Capability
    Raw packet queue pairs and flow steering need ``CAP_NET_RAW``. Give it to the binary after
    every build (``--dry-run`` needs it too, since the stages create their queue pair when they
    are constructed):

    .. code:: bash

         sudo setcap cap_net_raw+ep <build>/kotekan/kotekan

    A sudoers rule for that one command keeps rebuilds painless, e.g. in
    ``/etc/sudoers.d/kotekan-setcap``::

         user ALL=(root) NOPASSWD: /usr/sbin/setcap cap_net_raw+ep /path/to/build/kotekan/kotekan

Huge pages
    Give the payload buffers ``use_hugepages: true`` and reserve 2 MB pages for them on the NIC's
    NUMA node: ``num_frames`` times the frame size rounded up to 2 MB, per buffer. Fewer, larger
    pages mean fewer NIC address translations.

Locked memory
    Registering a frame with the NIC pins it. ``ulimit -l`` must cover every registered buffer.

MTU
    The interface MTU must be at least ``header_size + payload_size - 14``, e.g. 6194 for the
    CRS-sized packets in the example configs: ``sudo ip link set <interface> mtu 9000``.

Steering
    ``ibvUdpRecv`` steers every packet for its (destination MAC, IP, UDP port) to its queue pair.
    Those packets never reach the kernel, so nothing else on the host sees them, and the port
    should not be in use by another program. Other traffic on the interface is unaffected.

Packet format
    Header slots hold the whole on-wire header (Ethernet, IPv4, UDP and whatever application
    header follows), payload slots the UDP payload after that; ``lib/utils/udpPacket.hpp``
    describes the layout. The NIC fills in the IP and UDP checksums on send and verifies them on
    receive.

Checking throughput
^^^^^^^^^^^^^^^^^^^

* ``raw_ethernet_bw`` from the ``perftest`` package measures what the link and the NIC's PCIe
  slot can carry, independent of kotekan. Run it first: it is the ceiling for the stages.
* ``config/tests/ibv_udp_send.yaml`` and ``config/tests/ibv_udp_recv.yaml`` run a generator
  and a checker through the stages; ``tests/ibverbs/run_line_rate.sh`` starts both and prints
  what was sent, received and lost.
* The stages export Prometheus counters, ``curl -s localhost:12048/metrics | grep ibv_udp``.
  Packets the NIC dropped because no receive work request was posted appear in
  ``kotekan_ibv_udp_recv_out_of_buffer`` (device-wide) and in ``ethtool -S <interface>`` as
  ``rx_out_of_buffer``; raise ``queue_depth`` or ``buffer_depth``, pin the receive stage with
  ``cpu_affinity``, or switch ``poll_mode`` to ``busy`` if they grow.
* Between a ConnectX-6 Dx and a ConnectX-7 with 6208 byte packets, both poll modes receive
  100 Gb/s on the wire (2.0 Mpps) without loss. ``event`` costs the receive thread about a
  fifth of a core and ``busy`` a whole one; the sender needs about a tenth of a core.
