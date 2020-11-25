/**
 * @file
 * @brief The core of the DPDK packet capture code in kotekan
 *  - dpdkRXhandler
 *  - dpdkCore
 */

#ifndef DPDK_BASE_HPP
#define DPDK_BASE_HPP

// DPDK!
extern "C" {
#include <rte_ethdev.h> // for rte_eth_conf
#include <stdint.h>     // for uint32_t, int32_t, uint8_t
}

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for kotekanLogging

#include <exception> // for exception
#include <string>    // for string, allocator
#include <vector>    // for vector

/**
 * @brief Abstract object for processing packets that come from a given NIC port
 *
 * Implement a subclass of this object to make your own packet handlers
 *
 * @author Andre Renard
 */
class dpdkRXhandler : public kotekan::kotekanLogging {
public:
    /**
     * @brief Default constructor for the handler.
     *
     * This should be called by any subclass implementaion,
     * with a constructor that takes the same arugments
     *
     * @param config A reference to the config
     * @param unique_name The unique name of the handler, set by the config path.
     * @param buffer_container The container with all the named buffers
     * @param port The NIC port which this hander is attached to
     */
    dpdkRXhandler(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container, int port) :
        config(config),
        unique_name(unique_name),
        buffer_container(buffer_container),
        port(port) {

        set_log_level(config.get<std::string>(unique_name, "log_level"));
    };

    /**
     * @brief Default virtual destructor.
     */
    virtual ~dpdkRXhandler(){};

    /**
     * @brief Abstract function which is called each time a new packet comes in from the NIC
     *
     * Implement your own packet processing in the subclass of this object.
     *
     * @param mbuf Pointer to the DPDK rte_mbuf object containing the packet
     * @return int This function should return 0 if the packet was handled correctly.
     *             And return any other value if a critical error was encountered
     *             which requires the system shutdown.
     */
    virtual int handle_packet(struct rte_mbuf* mbuf) = 0;

    /**
     * @brief Called every 1 second to update stats
     *
     * Implement any stat updates, like prometheus metrics, here.
     */
    virtual void update_stats() = 0;

    // Allow internal access to handlers from dpdkCore
    friend class dpdkCore;

protected:
    /// The system config
    kotekan::Config& config;

    /// The unique name of this handler
    std::string unique_name;

    /// The container of buffers
    kotekan::bufferContainer& buffer_container;

    /// The NIC port which this handler is attached to.
    uint32_t port;
};

/**
 * @brief The core interface between DPDK enabled NICs and the kotekan framework.
 *
 * The idea of this objection is to deal with all the boiler plate which is needed
 * to setup DPDK on enabled NICs and map all the ports to lcores (DPDK threads).
 *
 * In the end anyone should be able to use this class with their own subclass of the
 * @c dpdkRXhandler without understanding all the details about setting up the DPDK framework.
 *
 * @conf   lcore_cpu_map    Array of CPU IDs which should be used for lcores (DPDK threads locked
 *                          to a CPU core).  For example [0,6] would create 2 lcores mapped
 *                          to the 1st and 7th CPU core.
 * @conf   lcore_port_map   Array of arrays mapping ports to lcores (DPDK threads locked to CPU
 *                          code) Format is index = lcore, value = array of port IDs
 *                          so @c [[0,1],[2,3]] maps lcore 0 to service ports 0 and 1,
 *                          and lcore 1 to service ports 2 and 3.
 *                          Note there is aways one handler per port, so that means
 *                          there can be more than one handler per lcore.
 * @conf   handlers         Array of json objections which each contain the config
 *                          line @c dpdk_handler:\<handler_name\> which names the hander
 *                          to use for the NIC port at its index in the handlers array.
 *                          Addational config for each handler can be given within each of
 *                          these objects.  For example:
 *                          handlers:
 *                             - dpdk_handler: myHandler # Handler for port 0
 *                                in_buf: my_buf_0
 *                              - dpdk_handler: myHandler # Handler for port 1
 *                                in_buf: my_buf_1
 *                          Note that if a port isn't being used it must be denoted by
 *                          `- dpdk_handler: none`.   The number of handlers much match the number
 *                          of ports in the system, even if they aren't being used by the current
 *                          config. There must be a valid handler for every port
 *                          referenced in @c lcore_port_map
 * @conf   master_lcore_cpu The CPU ID of the master lcore (which just handles simple things like
 *                          updating stats, and other low volume operatings)
 *
 * @par Optional config, don't change unless you know what you are doing.
 * @conf   num_mbufs        Int. Default 1024  The size of the mbuf pool
 * @conf   mbuf_cache_size  Int. Default 250   The number of mbufs to cache
 *                                             Basically this is to try and keep mbufs always in l3
 *                                             by reducing the number of mbufs used by default.
 * @conf   burst_size       Int. Default 32    The maximum number of packets returned by
 *                                             @c rte_eth_rx_burst
 * @conf   rx_ring_size     Int. Default 512   The size of the Receive ring
 * @conf   tx_ring_size     Int. Default 512   The size of the Transmit ring
 * @conf   max_rx_pkt_len   Int. Default 5000  The max packet size.
 * @conf   jumbo_frame      Bool. Default true Enable support for Jumbo frames
 * @conf   num_mem_channels Int. Default 4     The number of system memory channels
 * @conf   init_mem_alloc   Int.  Default 256  The initial memory allocation in MB
 *
 * @author Andre Renard
 */
class dpdkCore : public kotekan::Stage {
public:
    dpdkCore(kotekan::Config& config, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);
    ~dpdkCore();

    void main_thread() override;

    virtual std::string dot_string(const std::string& prefix) const override;

private:
    /**
     * @brief The actual DPDK runtime lcore function
     *
     * @param args A pointer to the dpdkCore object
     * @return int Always returns 0
     */
    static int lcore_rx(void* args);

    /**
     * @brief Starts the DPDK framework (ELA)
     *
     * @param lcore_cpu_map The mapping of lcores to CPU cores
     * @param master_lcore_cpu The master core CPU ID to bind too
     */
    void dpdk_init(std::vector<int> lcore_cpu_map, uint32_t master_lcore_cpu);

    /**
     * @brief Sets up a port for use with DPDK
     *
     * @param port The port ID to setup.
     * @param lcore_id The lcore which the port is being handled by.
     * @return  0 if it worked, and an error value if it failed.
     */
    int32_t port_init(uint8_t port, uint32_t lcore_id);

    /**
     * @brief Creates the handers for the ports
     *
     * @param buffer_container The buffer contain to pass onto the handlers.
     */
    void create_handlers(kotekan::bufferContainer& buffer_container);

    /// The pool of DPDK mbufs, one per numa node
    std::vector<struct rte_mempool*> mbuf_pools;

    /// Internal DPDK configuration struct
    struct rte_eth_conf port_conf;

    /// Number of lcores to run.
    uint32_t num_lcores;

    /// Number of ports handled
    uint32_t num_ports;

    /// The number of ports the system has (can be different from num_ports)
    uint32_t num_system_ports;

    /// The maximum number of packets returned by @c rte_eth_rx_burst
    uint32_t burst_size;

    /// The size of the Receive ring
    uint32_t rx_ring_size;

    /// The size of the Transmit ring
    uint32_t tx_ring_size;

    /// Just a list of ports with the length stored with it.
    struct portList {
        uint32_t* ports;
        uint32_t num_ports;
    };

    /// One of these port list structs exists per lcore
    struct portList* lcore_port_list;

    /// Number of memory channels
    uint32_t num_mem_channels;

    /// Initial memory allocation in MB
    uint32_t init_mem_alloc;

    /// One of these exists per system port
    dpdkRXhandler** handlers;
};


#endif /* DPDK_BASE_HPP */
