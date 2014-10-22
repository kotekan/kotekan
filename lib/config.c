
#include <jansson.h>
#include <string.h>
#include <assert.h>

#include "config.h"
#include "errors.h"

int parse_processing_config(struct Config* config, struct json_t * json)
{
    int error = 0;
    json_t * product_remap;

    error = json_unpack(json, "{s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:o}",
        "num_elements", &config->processing.num_elements,
        "num_local_freq", &config->processing.num_local_freq,
        "num_total_freq", &config->processing.num_total_freq,
        "samples_per_data_set", &config->processing.samples_per_data_set,
        "num_data_sets", &config->processing.num_data_sets,
        "buffer_depth", &config->processing.buffer_depth,
        "num_gpu_frames", &config->processing.num_gpu_frames,
        "product_remap", &product_remap);

    if (error) {
        ERROR("Error parsing processing config.");
        return error;
    }

    int remap_size = json_array_size(product_remap);

    if (remap_size != config->processing.num_elements) {
        ERROR("The remap array must have the same size as the number of elements.");
        return -2;
    }

    config->processing.product_remap = malloc(remap_size * sizeof(int));
    assert(config->processing.product_remap != NULL);

    for(int i = 0; i < remap_size; ++i) {
        config->processing.product_remap[i] = json_integer_value(json_array_get(product_remap, i));
    }

    // Special case for 16-element version
    if (config->processing.num_elements < 32) {
        config->processing.num_adjusted_elements = 32;
        config->processing.num_adjusted_local_freq = 64;
    } else {
        config->processing.num_adjusted_elements = config->processing.num_elements;
        config->processing.num_adjusted_local_freq = config->processing.num_local_freq;
    }

    // TODO Hard coded constants are BAD!
    config->processing.num_blocks = (config->processing.num_adjusted_elements / 32) *
        (config->processing.num_adjusted_elements / 32 + 1) / 2.;

    return 0;
}

int parse_ch_master_networking_config(struct Config* config, struct json_t * json)
{
    int error = 0;
    char * collection_server_ip;

    error = json_unpack(json, "{s:s, s:i}",
        "collection_server_ip", &collection_server_ip,
        "collection_server_port", &config->ch_master_network.collection_server_port);

    if (error) {
        ERROR("Error parsing ch_master_network config, check config file");
        return error;
    }

    if (collection_server_ip == NULL) {
        ERROR("The collection server IP address in the config file is not a valid string.");
        return -1;
    }

    config->ch_master_network.collection_server_ip = strdup(collection_server_ip);

    return 0;
}

int parse_gpu_config(struct Config* config, struct json_t * json)
{
    int error = 0;
    json_t * kernels;

    error = json_unpack(json, "{s:o, s:i, s:i}",
        "kernels", &kernels,
        "num_gpus", &config->gpu.num_gpus,
        "block_size", &config->gpu.block_size);

    if (error) {
        ERROR("Error parsing gpu config");
        return error;
    }

    config->gpu.num_kernels = json_array_size(kernels);
    if (config->gpu.num_kernels <= 0) {
        ERROR("No kernel file names given");
        return error;
    }
    config->gpu.kernels = malloc(config->gpu.num_kernels * sizeof(char *));
    assert(config->gpu.kernels != NULL);

    for (int i = 0; i < config->gpu.num_kernels; ++i) {
        const char * kernel_file_name = json_string_value(json_array_get(kernels, i));

        if (kernel_file_name == NULL) {
            ERROR("A kernel file name was invalid, or could not be parsed, check config file");
            return -1;
        }
        // Make a copy of the string for the config object.
        config->gpu.kernels[i] = strdup(kernel_file_name);
    }

    return 0;
}

int parse_fpga_network_config(struct Config* config, struct json_t * json)
{
    int error = 0;
    json_t * link_map;

    error = json_unpack(json, "{s:i, s:i, s:i, s:i, s:i, s:o}",
        "num_links", &config->fpga_network.num_links,
        "port_number", &config->fpga_network.port_number,
        "timesamples_per_packet", &config->fpga_network.timesamples_per_packet,
        "udp_frame_size", &config->fpga_network.udp_frame_size,
        "udp_packet_size", &config->fpga_network.udp_packet_size,
        "link_map", &link_map);

    if (error) {
        ERROR("Error parsing fpga_network config section.");
        return -1;
    }

    int num_links_in_map = json_array_size(link_map);
    if (num_links_in_map != config->fpga_network.num_links) {
        ERROR("The size of the link map must be equal to the number of links");
        return -2;
    }

    config->fpga_network.link_map = malloc(num_links_in_map * sizeof(struct LinkMap));
    assert(config->fpga_network.link_map != NULL);

    for (int i = 0; i < num_links_in_map; ++i) {
        json_t * link;
        char * link_name;

        link = json_array_get(link_map, i);
        if (link == NULL) {
            ERROR("Error reading link_map check config file");
            return -2;
        }

        error = json_unpack(link, "{s:s, s:i, s:i}",
            "link_name", &link_name,
            "gpu_id", &config->fpga_network.link_map[i].gpu_id,
            "link_id", &config->fpga_network.link_map[i].link_id);

        if (link_name == NULL) {
            ERROR("Link name is null for link %d in config file.", i);
            return -2;
        }

        config->fpga_network.link_map[i].link_name = strdup(link_name);
        config->fpga_network.link_map[i].stream_id = 0;
    }

    return 0;
}

int parse_config(struct Config* config, json_t * json)
{
    int error = 0;

    char * type;
    json_unpack(json, "{s:s}", "type", &type);

    if (strcmp(type, "config") != 0) {
        ERROR("The json object isn't a config object");
        return -1;
    }

    json_t * gpu_json, * fpga_network_json, * processing_json, * ch_master_network_json;
    error = json_unpack(json, "{s:o, s:o, s:o, s:o}", "gpu", &gpu_json, "fpga_network", &fpga_network_json,
        "processing", &processing_json, "ch_master_network", &ch_master_network_json);

    if (error) {
        ERROR("Error processing config root structure");
        return error;
    }

    error |= parse_processing_config(config, processing_json);
    error |= parse_ch_master_networking_config(config, ch_master_network_json);
    error |= parse_gpu_config(config, gpu_json);
    error |= parse_fpga_network_config(config, fpga_network_json);

    return error;
}

void delete_config(struct Config* config)
{

}

int load_config_from_file(struct Config* config, char* file_name)
{
    json_t *json;
    json_error_t json_error;
    int error;

    json = json_load_file(file_name, 0, &json_error);
    if (!json) {
        ERROR("The config file could not be read, or is invalid json.");
        return -1;
    }

    error = parse_config(config, json);

    return error;
}

void print_config(struct Config* config)
{
    // CH_MASTER Network Section
    INFO("config.ch_master_network.collection_server_ip = %s",
         config->ch_master_network.collection_server_ip);
    INFO("config.ch_master_network.collection_server_port = %d",
         config->ch_master_network.collection_server_port);

    // GPU section
    INFO("config.gpu.num_gpus = %d", config->gpu.num_gpus);
    INFO("config.gpu.block_size = %d", config->gpu.block_size);
    INFO("config.gpu.num_kernels = %d", config->gpu.num_kernels);
    for (int i = 0; i < config->gpu.num_kernels; ++i) {
        INFO("config.gpu.kernels[%d] = %s", i, config->gpu.kernels[i]);
    }

    // Processing section
    INFO("config.processing.num_elements = %d", config->processing.num_elements);
    INFO("config.processing.num_local_freq = %d", config->processing.num_local_freq);
    INFO("config.processing.num_total_freq = %d", config->processing.num_total_freq);
    INFO("config.processing.samples_per_data_set = %d", config->processing.samples_per_data_set);
    INFO("config.processing.num_data_sets = %d", config->processing.num_data_sets);
    INFO("config.processing.buffer_depth = %d", config->processing.buffer_depth);
    INFO("config.processing.num_adjusted_elements = %d", config->processing.num_adjusted_elements);
    INFO("config.processing.num_adjusted_local_freq = %d", config->processing.num_adjusted_local_freq);
    INFO("config.processing.num_blocks = %d", config->processing.num_blocks);

    for (int i = 0; i < config->processing.num_elements; ++i) {
        INFO("config.processing.product_remap[%d] = %d", i, config->processing.product_remap[i]);
    }

    // FPGA Network Section
    INFO("config.fpga_network.num_links = %d", config->fpga_network.num_links);
    INFO("config.fpga_network.port_number = %d", config->fpga_network.port_number);
    INFO("config.fpga_network.timesamples_per_packet = %d", config->fpga_network.timesamples_per_packet);
    INFO("config.fpga_network.udp_packet_size = %d", config->fpga_network.udp_packet_size);
    INFO("config.fpga_network.udp_frame_size = %d", config->fpga_network.udp_frame_size);

    for (int i = 0; i < config->fpga_network.num_links; ++i) {
        INFO("config.fpga_network.link_map[%d].link_name = %s", i,
             config->fpga_network.link_map[i].link_name);
        INFO("config.fpga_network.link_map[%d].gpu_id = %d", i,
             config->fpga_network.link_map[i].gpu_id);
        INFO("config.fpga_network.link_map[%d].stream_id = %d", i,
             config->fpga_network.link_map[i].stream_id);
        INFO("config.fpga_network.link_map[%d].link_id = %d", i,
             config->fpga_network.link_map[i].link_id);
    }
}

int num_links_in_group(struct Config* config, const unsigned int link_id)
{
    assert(link_id < config->fpga_network.num_links);

    int gpu_id = config->fpga_network.link_map[link_id].gpu_id;
    int num_links = 0;

    for (int i = 0; i < config->fpga_network.num_links; ++i) {
        if (config->fpga_network.link_map[i].gpu_id == gpu_id)
            num_links++;
    }

    return num_links;
}

int num_links_per_gpu(struct Config* config, const unsigned int gpu_id)
{
    assert(gpu_id < config->gpu.num_gpus);

    int num_links = 0;

    for (int i = 0; i < config->fpga_network.num_links; ++i) {
        if (config->fpga_network.link_map[i].gpu_id == gpu_id)
            num_links++;
    }

    return num_links;
}
