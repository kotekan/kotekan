#ifndef CONFIG_H
#define CONFIG_H

#include <jansson.h>

/// @brief Maps the a link to a GPU and set of buffers.
struct LinkMap {
    // The interface name of the link.
    char * link_name;

    // The GPU which processes data from this link.
    int gpu_id;

    // The stream/frequency ID assoicated with the link (not yet implemented).
    int stream_id;

    // The id of the link within a group
    int link_id;
};

/// @brief Values related to the overall GPU configuration.
struct GPUConfig {

    // The number of GPUs
    int num_gpus;

    // The block size.
    int block_size;

    // The number of kernels to compile.
    int num_kernels;

    // Kernels
    char ** kernels;

};

/// @brief Values related to the processing performed (e.g. intergration time, etc.).
struct ProcessingConfig {

    // The number of elements.
    // (an element is one ADC input, i.e. one polorization of a dual-pol feed).
    int num_elements;

    // The number of frequencies being processed by a given kernel
    int num_local_freq;

    // The total number of frequencies in the system
    int num_total_freq;

    // The number of samples per gpu data set (i.e. intergration period).
    int samples_per_data_set;

    // The number of data sets created by each kernel run.
    int num_data_sets;

    // Buffer Depth
    // TODO break this up into more than one global depth
    int buffer_depth;

    // The remapping of products from FPGA order to "physical (on the focal line) order"
    int * product_remap;

    /// NOTE The next two values are specal cases when the number of elements is < 32, i.e.
    /// when the correlator is running in 16-element mode with one FPGA.

    // This value is only different if num_elements < 32, otherwise it is identical to num_elements.
    int num_adjusted_elements;

    // This value is only different if num_elements < 32, otherwise it is identical to num_local_freq;
    int num_adjusted_local_freq;

    // The number of blocks required in the kernel (based on the number of elements and the block size).
    int num_blocks;
};

/// @brief Struct for information about the FPGA data streams
struct FPGANetworkConfig {

    // The number of FPGA data lines being processed.
    int num_links;

    // The mapping of links to GPUs.
    struct LinkMap * link_map;

    // The port number to receive packets on.
    int port_number;

    // The number of frames (time samples) in each packet.
    int timesamples_per_packet;

    // The total size of the UDP packet, including the ethernet frame headers and footers.
    int udp_packet_size;

    // The size of the data in each frame.
    int udp_frame_size;
};

/// @brief Sturct for information about ch_master related configuration.
struct CH_MASTER_NetworkConfig {

    // The IP address of the collection server.
    char * collection_server_ip;

    // The TCP port to use on the collection server.
    int collection_server_port;
};

/// @brief Struct for holding static system configuration
/// This struct should not be used for values which change in the course of
/// operation - and should only change when system resets.
struct Config {
    /// ch_master (server) configuration
    struct CH_MASTER_NetworkConfig ch_master_network;

    /// GPU configuration (number of GPUs, etc.)
    struct GPUConfig gpu;

    /// Network configuration
    struct FPGANetworkConfig fpga_network;

    /// Data processing configuration
    struct ProcessingConfig processing;
};

/// @brief Parses a json object which contains the configuration for kotekan
/// @param config A reference to the config struct to be populated.
/// @param json A pointer to a json object with the config data
/// @return 0 on success, else failure code.
int parse_config(struct Config* config, json_t * json);

/// @brief Loads a json config from file.
/// @param config A reference to the config struct to be populated.
/// @param file_name The name of the file to load.
/// @return 0 on success, else failure code.
int load_config_from_file(struct Config * config, char * file_name);

/// @brief Dumps the config struct containts to the log system
/// @param config The config struct to dump
void print_config(struct Config * config);

/// @brief cleans up memory used by the config struct.
void delete_config(struct Config * config);

// Functions for derived data.

/// @brief Returns the number of links assigned to a gpu based on the link ID
int num_links_in_group(struct Config * config, const unsigned int link_id);

/// @brief Returns the number of links assigned to a gpu based on the gpu ID
int num_links_per_gpu(struct Config * config, const unsigned int gpu_id);

#endif