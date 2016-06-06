#ifndef CONFIG_H
#define CONFIG_H

#include <jansson.h>

#ifdef __cplusplus
extern "C" {
#endif

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

    // ** Time shift options **

    // Use time shift kernel?
    int use_time_shift;

    // The first element (in FPGA order) to shift.
    int ts_element_offset;

    // The number of elements to shift starting at ts_element_offset.
    int ts_num_elem_to_shift;

    // The number of timesamples to shift above elements.
    int ts_samples_to_shift;

    // ** Beamforming options **

    // Do beamforming
    int use_beamforming;

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

    // The number of GPU frames to add before sending to ch_master
    int num_gpu_frames;

    // The remapping of products from FPGA order to "physical (on the focal line) order"
    int * product_remap;

    int * inverse_product_remap;

    // Data limit, used for testing to take "snapshots"
    int data_limit;

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

    // Disable upload
    int disable_upload;
};

struct Beamforming {
    // The IP address of the collection server
    char * vdif_server_ip;

    // The port to send packets to.
    int vdif_port;

    // The latitude and longitude of the instrument
    double instrument_lat, instrument_long;

    // The ra and dec the pointing should be set to.
    // NOTE: This will be made more complex to allow more than one pointing in a run.
    double ra, dec;

    // The positions of the feeds.
    float * element_positions;

    // Number of masked elements
    int num_masked_elements;

    // The array of elements to mask out of the beamforming.
    int * element_mask;

    // The gain amount.
    double scale_factor;

    // Do not track. ( 0 => track, 1 => do not track source )
    int do_not_track;

    // Option used if do_not_track is set to 1.
    // 0 means use current time
    // otherwise this is the unix time stamp to point to.
    int fixed_time;
};

struct RawCapture {

    int enabled;
    int num_disks;
    char * disk_base;
    char * disk_set;
    char * note;
    char * ram_disk_dir;
    char * instrument_name;
    int write_packets;
    int write_powers;
    int legacy_power_output;
    int samples_per_file;
    int stream_vdif;
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

    /// The beamforming options
    struct Beamforming beamforming;

    /// The raw capture options.
    struct RawCapture raw_cap;
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

#ifdef __cplusplus
}
#endif

#endif
